import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import numpy as np


def train(args, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_1 = AverageMeter()  # loss (per word decoded) for language L1
    top5accs_1 = AverageMeter()  # top5 accuracy for language L1
    losses_2 = AverageMeter()  # loss (per word decoded) for language L2
    top5accs_2 = AverageMeter()  # top5 accuracy for language L1

    start = time.time()

    # Batches
    for i, (imgs, caps1, caplens1, caps2, caplens2) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        
        caps1 = caps1.to(device)
        caplens1 = caplens1.to(device)
        caps2 = caps2.to(device)
        caplens2 = caplens2.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        
        # imgs: [batch_size, 14, 14, 2048]
        # caps: [batch_size, 52]
        # caplens: [batch_size, 1]
        scores1, caps_sorted1, decode_lengths1, alphas1, sort_ind1 = decoder(imgs, caps1, caplens1, args.language1)
        scores2, caps_sorted2, decode_lengths2, alphas2, sort_ind2 = decoder(imgs, caps2, caplens2, args.language2)
        
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets1 = caps_sorted1[:, 1:]
        targets2 = caps_sorted2[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores1 = pack_padded_sequence(scores1, decode_lengths1, batch_first=True).data
        targets1 = pack_padded_sequence(targets1, decode_lengths1, batch_first=True).data
        
        scores2 = pack_padded_sequence(scores2, decode_lengths2, batch_first=True).data
        targets2 = pack_padded_sequence(targets2, decode_lengths2, batch_first=True).data
        # print(scores.size())
        # print(targets.size())

        # Calculate loss
        loss1 = criterion(scores1, targets1)
        loss2 = criterion(scores2, targets2)
        loss = loss1 + loss2
        # Add doubly stochastic attention regularization
        # Second loss, mentioned in paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
        # https://arxiv.org/abs/1502.03044
        # In section 4.2.1 Doubly stochastic attention regularization: We know the weights sum to 1 at a given timestep.
        # But we also encourage the weights at a single pixel p to sum to 1 across all timesteps T.
        # This means we want the model to attend to every pixel over the course of generating the entire sequence.
        # Therefore, we want to minimize the difference between 1 and the sum of a pixel's weights across all timesteps.
        
        loss += args.alpha_c * ((1. - alphas1.sum(dim=1)) ** 2).mean()
        loss += args.alpha_c * ((1. - alphas2.sum(dim=1)) ** 2).mean()


        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5_1 = accuracy(scores1, targets1, 5)
        losses_1.update(loss1.item(), sum(decode_lengths1))
        top5accs_1.update(top5_1, sum(decode_lengths1))
        
        top5_2 = accuracy(scores2, targets2, 5)
        losses_2.update(loss2.item(), sum(decode_lengths2))
        top5accs_2.update(top5_2, sum(decode_lengths2))
        
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss 1: {} AVG_Loss 1: {} Top-5 Accuracy 1: {} Loss 2: {} AVG_Loss 2: {} Top-5 Accuracy 2: {} Batch_time: {}s".format(epoch+1, args.epochs, i+1, len(train_loader), losses_1.val, losses_1.avg, top5accs_1.val, losses_2.val, losses_2.avg, top5accs_2.val, batch_time.val))


def validate(args, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses_1 = AverageMeter()  # loss (per word decoded)
    top5accs_1 = AverageMeter()  # top5 accuracy
    losses_2 = AverageMeter()  # loss (per word decoded)
    top5accs_2 = AverageMeter()  # top5 accuracy


    start = time.time()

    references1 = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses1 = list()  # hypotheses (predictions)
    
    references2 = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses2 = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, (imgs, caps1, caplens1, allcaps1, caps2, caplens2, allcaps2) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps1 = caps1.to(device)
            caplens1 = caplens1.to(device)
            caps2 = caps2.to(device)
            caplens2 = caplens2.to(device)


            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores1, caps_sorted1, decode_lengths1, alphas1, sort_ind1 = decoder(imgs, caps1, caplens1, args.language1)
            scores2, caps_sorted2, decode_lengths2, alphas2, sort_ind2 = decoder(imgs, caps2, caplens2, args.language2)


            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets1 = caps_sorted1[:, 1:]
            targets2 = caps_sorted2[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy1 = scores1.clone()
            scores_copy2 = scores2.clone()
            
            scores1 = pack_padded_sequence(scores1, decode_lengths1, batch_first=True).data
            targets1 = pack_padded_sequence(targets1, decode_lengths1, batch_first=True).data

            scores2 = pack_padded_sequence(scores2, decode_lengths2, batch_first=True).data
            targets2 = pack_padded_sequence(targets2, decode_lengths2, batch_first=True).data

            # Calculate loss
            loss1 = criterion(scores1, targets1)
            loss2 = criterion(scores2, targets2)
            loss = loss1 + loss2

            # Add doubly stochastic attention regularization
           
            loss += args.alpha_c * ((1. - alphas1.sum(dim=1)) ** 2).mean()
            loss += args.alpha_c * ((1. - alphas2.sum(dim=1)) ** 2).mean()
            
            # Keep track of metrics
            top5_1 = accuracy(scores1, targets1, 5)
            losses_1.update(loss1.item(), sum(decode_lengths1))
            top5accs_1.update(top5_1, sum(decode_lengths1))

            top5_2 = accuracy(scores2, targets2, 5)
            losses_2.update(loss2.item(), sum(decode_lengths2))
            top5accs_2.update(top5_2, sum(decode_lengths2))
        
            batch_time.update(time.time() - start)
            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps1 = allcaps1[sort_ind1]  # because images were sorted in the decoder
            for j in range(allcaps1.shape[0]):
                img_caps = allcaps1[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map1['<start>'], word_map1['<pad>']}],
                        img_caps))  # remove <start> and pads
                references1.append(img_captions)
                
            allcaps2 = allcaps2[sort_ind2]  # because images were sorted in the decoder
            for j in range(allcaps2.shape[0]):
                img_caps = allcaps2[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map2['<start>'], word_map2['<pad>']}],
                        img_caps))  # remove <start> and pads
                references2.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy1, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths1[j]])  # remove pads
            preds = temp_preds
            hypotheses1.extend(preds)
            
            _, preds = torch.max(scores_copy2, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths2[j]])  # remove pads
            preds = temp_preds
            hypotheses2.extend(preds)

            assert len(references1) == len(hypotheses1)
            assert len(references2) == len(hypotheses2)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    metrics = get_eval_score(references1, hypotheses1)

    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} ROUGE_L {} CIDEr {}".format
          (losses_1.avg, top5accs_1.avg,  metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],\
           metrics["ROUGE_L"], metrics["CIDEr"]))
    
    metrics2 = get_eval_score(references2, hypotheses2)

    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} ROUGE_L {} CIDEr {}".format
          (losses_2.avg, top5accs_2.avg,  metrics2["Bleu_1"],  metrics2["Bleu_2"],  metrics2["Bleu_3"],  metrics2["Bleu_4"],\
           metrics2["ROUGE_L"], metrics2["CIDEr"]))

    return metrics, metrics2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    # Data parameters
    parser.add_argument('--data_folder', default="./dataset/generated_data",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="multi30k_5_cap_per_img_3_min_word_freq",
                        help='base name shared by data files.')
    parser.add_argument('--save_name', default="",
                        help='save the model.')
    parser.add_argument('--language1', default="EN", help='language for the first decoder')
    parser.add_argument('--language2', default="DE", help='language for the second decoder')
    parser.add_argument('--cpi', type=int, default=5, help='captions per image.')
    
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=10, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=500, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=1, help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=False, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    
    
    
    args = parser.parse_args()

    
    language_pair = [args.language1, args.language2]
    # the first decoder must generate English captions
    assert language_pair[0] == 'EN'

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim,
                 "attention_dim": args.attention_dim,
                 "decoder_dim": args.decoder_dim,
                 "dropout": args.dropout}

    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    
    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + language_pair[0] + '_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map1 = json.load(j)

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + language_pair[1] + '_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map2 = json.load(j)


    # Initialize / load checkpoint
    if args.checkpoint is None:
        encoder = CNN_Encoder()
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None

        
        decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=[len(word_map1),len(word_map2)],
                                       dropout=args.dropout)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['metrics'][0]["Bleu_4"] + checkpoint['metrics'][1]["Bleu_4"]
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        decoder.fine_tune_embeddings(args.fine_tune_embedding)
        # load final_args from checkpoint
        final_args = checkpoint['final_args']
        for key in final_args.keys():
            args.__setattr__(key, final_args[key])
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            print("Encoder_Optimizer is None, Creating new Encoder_Optimizer!")
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    train_data = CaptionDataset(args.data_folder, args.data_name, 'TRAIN', languages = language_pair,\
                                       transform=transforms.Compose([normalize]))
    train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    val_data = CaptionDataset(args.data_folder, args.data_name, 'VAL', languages = language_pair,\
                              transform=transforms.Compose([normalize]))
    val_loader = torch.utils.data.DataLoader(val_data,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Epochs

    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 5 consecutive epochs
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder and encoder_optimizer is not None:
                print(encoder_optimizer)
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(args, train_loader=train_loader, encoder=encoder, decoder=decoder, criterion=criterion,
              encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, epoch=epoch)

        # One epoch's validation
        metrics,metrics2 = validate(args, val_loader=val_loader, encoder=encoder, decoder=decoder, criterion=criterion)
        recent_bleu4 = metrics["Bleu_4"] + metrics2['Bleu_4']

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.save_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, [metrics,metrics2] , is_best, final_args)
