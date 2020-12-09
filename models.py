import torch
from torch import nn
import torchvision
from collections import OrderedDict

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    Resnet50.
    """

    def __init__(self, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=True)
        # Remove last linear layer and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for backbone CNN.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children()):
            for p in c.parameters():
                p.requires_grad = fine_tune

class LF(nn.Module):
    def __init__(self, encoded_image_size=14):
    
        super(LF, self).__init__()

        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet50(pretrained = True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) 
        self.layers = [
            ('input_image', lambda x:x),
            ('conv1', lambda x: self.resnet[0](x)),
            ('conv5', lambda x: self.resnet[4](self.resnet[3](
                self.resnet[2](self.resnet[1](x))))),
            ('conv9',  lambda x: self.resnet[5](x)),
            ('conv13', lambda x: self.resnet[6](x)),
            ('conv17', lambda x: self.resnet[7](x)),
            ('adp_pool', lambda x: self.adaptive_pool(x)),
        ]
            
    def forward(self, x):
        for name, operator in self.layers:
            x = operator(x)
            setattr(self, name, x)
        # Take the max for each prediction map.
        return x.permute(0, 2, 3, 1)
    
    def partial_forward(self, start):
        skip = True
        for name, operator in self.layers:
            if name == start:
                x = getattr(self, name)
                skip = False
            elif skip:
                continue
            else:
                x = operator(x)
                setattr(self, name, x)

        return x.permute(0, 2, 3, 1)

    
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # [batch_size_t, num_pixels=196, 2048] -> [batch_size_t, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size_t, decoder_dim=512] -> [batch_size_t, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [batch_size_t, num_pixels=196, attention_dim] -> [batch_size_t, num_pixels]
        alpha = self.softmax(att)  # [batch_size_t, num_pixels=196]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size_t, encoder_dim=2048]

        return attention_weighted_encoding, alpha

    
class DecoderWithAttention(nn.Module):
    """
    Double-Decoder.
    Two seperate decoders share the same attention layer.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        
        assert(len(vocab_size)==2)
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size1 = vocab_size[0]
        self.vocab_size2 = vocab_size[1]
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        
        self.embedding = nn.Embedding(self.vocab_size1, embed_dim, padding_idx=0)  # embedding layer
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(decoder_dim, self.vocab_size1)  # linear layer to find scores over vocabulary
        
        self.dropout = nn.Dropout(p=self.dropout)
        self.sigmoid = nn.Sigmoid()
        
        self.embedding_sec = nn.Embedding(self.vocab_size2, embed_dim, padding_idx=0)  # embedding layer
        self.decode_step_sec = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h_sec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_sec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta_sec = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc_sec = nn.Linear(decoder_dim, self.vocab_size2)  # linear layer to find scores over vocabulary
        
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_tune_embeddings()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        self.embedding_sec.weight.data.uniform_(-0.1, 0.1)
        self.fc_sec.bias.data.fill_(0)
        self.fc_sec.weight.data.uniform_(-0.1, 0.1)


    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding_sec.weight = nn.Parameter(embeddings_sec)
        
    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
        for p in self.embedding_sec.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out, language):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  
        if language == 'EN':
            h = self.init_h(mean_encoder_out)  
            c = self.init_c(mean_encoder_out)
        elif language == 'DE':
            h = self.init_h_sec(mean_encoder_out)  
            c = self.init_c_sec(mean_encoder_out)
        else:
            print('Input language is not defined.')
            assert(0)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, language):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size1 = self.vocab_size1
        vocab_size2 = self.vocab_size2
        
        # Flatten image -> [batch_size, num_pixels=196, encoder_dim=2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        
        if len(caption_lengths.size()) == 1:
            caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        else:
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind] # sort encoded image based on length of caption 1 languegs

        
        encoded_captions = encoded_captions[sort_ind]

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        
        if language == 'EN':
            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out, language)  # [batch_size, decoder_dim]
            # Create tensors to hold word predicion scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size1).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
            
            # At each time-step, decode by
            # attention-weighing the encoder's output based on the decoder's previous hidden state output
            # then generate a new word in the decoder with the previous word and the attention weighted encoding
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
            
        elif language == 'DE':
        
            embeddings = self.embedding_sec(encoded_captions)
            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out, language)  # [batch_size, decoder_dim]

            # Create tensors to hold word predicion scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size2).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta_sec(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step_sec(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc_sec(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
        else:
            assert(0)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    