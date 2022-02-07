###################################
# Content of caption.py
###################################

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import easydict
import argparse
#from scipy.misc import imresize  # suppression de scipy.misc.imread qui n'existe plus
from PIL import Image
from imageio import imread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, cell_type, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)  # Provient de imageio, remplacé car scipy.misc.imread n'existe plus
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize(size=(256, 256)))   # Previously imresize(img, (256, 256)) with scipy.misc. Here shape (256, 256, 3)
    img = np.transpose(img,[2, 0, 1]) # Then shape (3, 256, 256)
    img = img / 255.  # Normalisation
    img = torch.FloatTensor(img).to(device)  # Torch Tensor format, to use GPU
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)  # 14 pr ResNet
    encoder_dim = encoder_out.size(3)  # 2048 pr ResNet
    # encoder_out.shape : (1, 14, 14, 2048)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim), 14x14 et 2048 pr ResNet
    num_pixels = encoder_out.size(1)  # 14x14

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)   # On répète donc k x la sortie de l'encoder précédamment calculé. On rappelle : k = beam_size

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1), chaque phrase commence par un <SOS> (Start of sentence)
    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        # We have to find the k best words for the k previously selected words (Concept of Beam Search)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)  # This is just a resize, to visualise the attention on the image

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)  # WHAT THE FUCK IS f_beta ?! Sure a gate but I haven't seen it in any course
        awe = gate * awe

        if cell_type=='LSTM':
          h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)  # On passe à travers la cell LSTM
        elif cell_type=='GRU':
          h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)  # On passe à travers la cell GRU
        else:
          return('Enter a valid RNN Cell name')

        scores = decoder.fc(h)  # (s, vocab_size)   
        scores = F.log_softmax(scores, dim=1)  # We compute the score to choose the k best words for this current-previously-selected word

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size) # For the k current Beam-sentences, we compute the scores to select the next words

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)  # And among the k*vocab_size possibilities, we choose the best k sentences.

        # Convert unrolled indices to actual indices of scores"
        prev_word_inds =  top_k_words // vocab_size # Error previously because the division wasn't floored like now (top_k_words / vocab_size  # (s))
        next_word_inds = top_k_words % vocab_size  # (s)
        print()
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    models_to_train = [['resnet', 'LSTM', False, 2048], ['resnet', 'GRU', False, 2048], ['vgg', 'LSTM', False, 512], ['resnet', 'LSTM', True, 2048]]
    for cnn_rnn_structure in models_to_train:
        #cnn_rnn_structure = models_to_train[0]
        cell_type = cnn_rnn_structure[1]
        image_number = '102351840_323e3de834'

        parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
        parser.add_argument('--img', '-i', default=r'/content/drive/MyDrive/GGColab/BE1_Image_Captionning/Flicker8k_Dataset/{}.jpg'.format(image_number), help='path to image')  # img 667626_18933d713e.jpg
        parser.add_argument('--word_map', '-wm', default=r'/content/drive/MyDrive/GGColab/BE1_Image_Captionning/prepared_data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json', help='path to word map JSON')
        parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
        parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
        parser.add_argument('--model', '-m', default=r'/content/drive/MyDrive/GGColab/BE1_Image_Captionning/BEST_{}_{}_{}_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'.format(cnn_rnn_structure[0], cnn_rnn_structure[1], cnn_rnn_structure[2]) , help='path to model')
        parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        args = parser.parse_args()
        
        # Load model
        checkpoint = torch.load(args.model, map_location=str(device))  # Replace 'load' by 'as_tensor'
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        # Load word map (word2ix)
        with open(args.word_map, 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, cell_type,args.beam_size)
        alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
