import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import argparse
import json

from nltk.translate.bleu_score import corpus_bleu
import pickle

import json
from tqdm import tqdm

import time
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *


import torch.optim as optim

  
def evaluate_lstm(beam_size, word_map, encoder_outs, allcaps, target, decoder, vocab_size):
    beam_size = beam_size
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    results = {}

   
    with torch.no_grad():
        enc_image_size = encoder_outs.size(1)
        encoder_dim = encoder_outs.size(-1)
        for id_ in range(0,encoder_outs.size(0)):
            k = beam_size
            encoder_out = encoder_outs[id_].unsqueeze(0).view(1, -1, encoder_dim)
            num_pixels = enc_image_size*enc_image_size
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0, never used
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            step = 1
            exceed = False
            h, c = decoder.init_hidden_state(encoder_out, target)

            # English
            while True:
                if target == 'EN':        
                    embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                    awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                    gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                    awe = gate * awe

                    h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                    scores = decoder.fc(h)  # (s, vocab_size)
                    scores = F.log_softmax(scores, dim=1)
                elif target == 'DE':
                    embeddings = decoder.embedding_sec(k_prev_words).squeeze(1)  # (s, embed_dim)

                    awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                    gate = decoder.sigmoid(decoder.f_beta_sec(h))  # gating scalar, (s, encoder_dim)
                    awe = gate * awe

                    h, c = decoder.decode_step_sec(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                    scores = decoder.fc_sec(h)  # (s, vocab_size)
                    scores = F.log_softmax(scores, dim=1)
                else:
                    print('Wrong target!')
                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]

                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    exceed=True
                    break
                step += 1
            if exceed:
                seq = seqs[0][:20].cpu().tolist()
#                 print('bad prediction for English, ', i)
            else:    
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
                
            img_caps = allcaps[id_].unsqueeze(0)[0].tolist()

            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps)) 
            references.append(img_captions)
            # Hypotheses
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            
            assert len(references) == len(hypotheses)
        return references, hypotheses


def feedback_val(args, val_loader, encoder, decoder, word_map_target, lr, layer, max_iter, criterion):
    references={}
    hypotheses={}
    t = tqdm(val_loader, desc="Evaluating on Val:")
    for i, (imgs, source_cap, source_caplen, source_cap_all, target_cap, target_caplen, target_cap_all) in enumerate(t):

        imgs = imgs.to(device)
        source_cap = source_cap.to(device)
        source_caplen = source_caplen.to(device)
        source_cap_all = source_cap_all.to(device)
        
        target_cap_all = target_cap_all.to(device)
        
        imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, source_cap, source_caplen, args.source)    
        
        activation = getattr(encoder, args.layer).data
        activation.requires_grad = True

        setattr(encoder, args.layer, activation)
        optimizer = optim.Adam([activation], lr = lr, weight_decay = 1e-4)
        
        
        for iteration in range(0, max_iter):
            output = encoder.partial_forward(args.layer)
            
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(output, source_cap, source_caplen, args.source)       
            targets = caps_sorted[:, 1:]
            
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            
            
            # get bleu-score using updated image features
            re, hy = evaluate_lstm(args.beam_size, word_map_target, output, target_cap_all, args.target, \
                               decoder,len(word_map_target))

            if iteration not in references:
                references[iteration] = re
                hypotheses[iteration] = hy


            else:
                references[iteration].extend(re)
                hypotheses[iteration].extend(hy)
        
            # update pivot                    
            loss = criterion(scores, targets) 
            loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
                       
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()


        activation.requires_grad = False 

    return references, hypotheses

def feedback_test(args, test_loader, encoder, decoder, word_map_target, lr, layer, best_iter, criterion):

    references={}
    hypotheses={}
    
    t = tqdm(test_loader, desc="Evaluating on Test:")
    for i, (imgs, source_cap, source_caplen, source_cap_all, target_cap, target_caplen, target_cap_all) in enumerate(t):
        
        imgs = imgs.to(device)
        source_cap = source_cap.to(device)
        source_caplen = source_caplen.to(device)
        source_cap_all = source_cap_all.to(device)
        
        # to evaluate target language
        target_cap_all = target_cap_all.to(device)
        
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, source_cap, source_caplen, args.source)    
        
        activation = getattr(encoder, args.layer).data
        activation.requires_grad = True

        setattr(encoder, args.layer, activation)

        optimizer = optim.Adam([activation], lr = lr, weight_decay = 1e-4)
        for iteration in range(0, best_iter+1):
            output = encoder.partial_forward(args.layer)
           
            if iteration == 0:
                
                re, hy = evaluate_lstm(args.beam_size, word_map_target, output.detach().clone(), \
                                             target_cap_all, args.target, decoder,len(word_map_target))

                if iteration not in references:
                    references[iteration] = re
                    hypotheses[iteration] = hy

                else:
                    references[iteration].extend(re)
                    hypotheses[iteration].extend(hy)
                    
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(output, source_cap, source_caplen, args.source)       
            targets = caps_sorted[:, 1:]
            
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets) 
            loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
                       
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
            
        activation.requires_grad = False 
        output = encoder.partial_forward(args.layer)

        
        re, hy = evaluate_lstm(args.beam_size, word_map_target, output, target_cap_all, args.target, \
                               decoder,len(word_map_target))
        
        if iteration > 0:
            if iteration not in references:
                references[iteration] = re
                hypotheses[iteration] = hy
                
            else:
                references[iteration].extend(re)
                hypotheses[iteration].extend(hy)
            
    return references, hypotheses 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    parser.add_argument('--data_folder', default="./dataset/generated_data",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="coco_5_cap_per_img_5_min_word_freq",
                        help='base name shared by data files.')
    parser.add_argument('--decoder_mode', default="lstm", help='which model does decoder use?') 
    parser.add_argument('--beam_size', type=int, default=3, help='beam size.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--checkpoint', default="./BEST_checkpoint.pth.tar",
                        help='model checkpoint.')
    parser.add_argument('--source', default="EN",
                        help='source language, [EN, DE].')
    parser.add_argument('--target', default="DE",
                        help='target language, [EN, DE, JP].')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate [Default: 1e-3].")
    parser.add_argument("--layer", type=str, default="conv13",
                        help="target language:[conv5|conv9|conv13|conv17]")
    parser.add_argument("--max_iter", type=int, default=20,
                        help="max iteration during validation [Default: 20].")
    parser.add_argument("--type", type=str, default="normal",
                        help="using real images or noise images: [normal, random]")
    
    parser.add_argument('--embed_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    args = parser.parse_args()
   
    assert args.source and args.target in ['EN', 'DE', 'JP']
    language = [args.source, args.target]
    print('Two languages are ', language)
    word_map_source_file = os.path.join(args.data_folder, 'WORDMAP_' + language[0] + '_' + args.data_name + '.json')
    word_map_target_file = os.path.join(args.data_folder, 'WORDMAP_' + language[1] + '_' + args.data_name + '.json')
    
    # Load word map (word2id)
    with open(word_map_source_file, 'r') as j:
        word_map_source = json.load(j)
    with open(word_map_target_file, 'r') as j:
        word_map_target = json.load(j)
    
    vocab_size = len(word_map_target)

    device = torch.device("cuda")
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    fixed_vocab_size = [len(word_map_source),len(word_map_target)] if args.source == 'EN' \
    else [len(word_map_target),len(word_map_source)]
    decoder = DecoderWithAttention(attention_dim=args.attention_dim, embed_dim=args.embed_dim, decoder_dim=args.decoder_dim, \
                                   vocab_size=fixed_vocab_size, dropout=args.dropout)
    decoder.load_state_dict(checkpoint['decoder'].state_dict())
    decoder = decoder.to(device)
    decoder.eval()
        
    encoder = LF(14)
    encoder.load_state_dict(checkpoint['encoder'].state_dict())
    encoder = encoder.to(device)
    encoder.eval()

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # load val/test data
    random = True if args.type == 'random' else False
    print('Feedback image type is noise', random)
    
    val_split = 'VAL'
    test_split = 'TEST'    
    val_dataset = CaptionDataset(args.data_folder, args.data_name, val_split, languages = [args.source,args.target],\
                              random = random, transform=transforms.Compose([normalize]))
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print('Validation data length is: ', len(val_dataset))
    
    test_dataset = CaptionDataset(args.data_folder, args.data_name, test_split, languages = [args.source,args.target],\
                              random = random, transform=transforms.Compose([normalize]))
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print('Testing data length is: ', len(test_dataset))
    
    # set up criterion
    criterion = nn.CrossEntropyLoss().to(device)
    
    bleu_list_val = [] 
      
    references_val, hypotheses_val = \
    feedback_val(args, val_loader, encoder, decoder, word_map_target, args.lr, args.layer, args.max_iter, criterion)
    
    for it in references_val:
        bleu_list_val.append(corpus_bleu(references_val[it], hypotheses_val[it]))
    print(bleu_list_val)
    print("val best iteration: ", bleu_list_val.index(max(bleu_list_val)))
    best_iter = bleu_list_val.index(max(bleu_list_val))
    
    encoder.eval()
    decoder.eval()

    references_test, hypotheses_test = feedback_test(args, test_loader, encoder, decoder, word_map_target, args.lr, args.layer,\
                                                     best_iter, criterion)
    print("Target language is: ", args.target)
    print("before:", corpus_bleu(references_test[0], hypotheses_test[0]))
    print("after:", corpus_bleu(references_test[best_iter], hypotheses_test[best_iter]))
    
