# coding: utf-8
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model as MODEL

import hydra
import logging
from torch.utils.tensorboard import SummaryWriter



@hydra.main(config_path="./cfg", config_name='config')
def main(cfg):
    step = 0
    orig_cwd = hydra.utils.get_original_cwd()
    
    curr_cwd = os.getcwd()
    logger = SummaryWriter(curr_cwd)
   
    
    #################################
    #################################
    #################################
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)


    ###############################################################################
    # Training code
    ###############################################################################

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(cfg.model.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        if cfg.model.name != 'Transformer':
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, cfg.model.bptt):
                data, targets = get_batch(data_source, i)
                if cfg.model.name == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)


    def train(step):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        if cfg.model.name != 'Transformer':
            hidden = model.init_hidden(cfg.model.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, cfg.model.bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if cfg.model.name == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            total_loss += loss.item()

            if batch % cfg.log_interval == 0 and batch > 0:
                cur_loss = total_loss / cfg.log_interval
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // cfg.model.bptt, lr,
                    elapsed * 1000 / cfg.log_interval, cur_loss, math.exp(cur_loss)))
                logger.add_scalar('Train Loss', cur_loss, step)
                step += cfg.log_interval
                total_loss = 0
                start_time = time.time()

            if cfg.dry_run:
                break
            


    def export_onnx(path, batch_size, seq_len):
        logging.info('The model is also exported in ONNX format at {}.'.format(os.path.realpath(cfg.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
        hidden = model.init_hidden(batch_size)
        torch.onnx.export(model, (dummy_input, hidden), path)

    ###################################
    ###################################
    ######## start main ###############
    ###################################
    

    # Set the random seed manually for reproducibility.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        if not cfg.cuda:
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    device = torch.device("cuda" if cfg.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    datapath = os.path.join(orig_cwd, cfg.data)
    corpus = data.Corpus(datapath)

    eval_batch_size = 10
    train_data = batchify(corpus.train, cfg.model.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    if cfg.model.name == 'Transformer':
        model = MODEL.TransformerModel(ntokens, cfg.model.emsize, cfg.model.nhead, cfg.model.nhid, cfg.model.nlayers, cfg.model.dropout, cfg.model.tied).to(device)
    else:
        model = MODEL.RNNModel(cfg.model.name, ntokens, cfg.model.emsize, cfg.model.nhid, cfg.model.nlayers, cfg.model.dropout, cfg.model.tied).to(device)

    criterion = nn.NLLLoss()

    logging.info ("Vocabulary Size: {}".format(ntokens))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info ("Total number of model parameters: {:.2f}M".format(num_params*1.0/1e6))


    # Loop over epochs.
    lr = cfg.model.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, cfg.model.epochs+1):
            epoch_start_time = time.time()
            train(step)
            val_loss = evaluate(val_data)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            logger.add_scalar('val loss', val_loss, epoch)
            logging.info('-' * 89)
            # Run on test data.
            test_loss = evaluate(test_data)
            logger.add_scalar('test loss', test_loss, epoch)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(cfg.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 3.0
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')

    # Load the best saved model.
    with open(cfg.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if cfg.model.name in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    logging.info('=' * 89)
    logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logging.info('=' * 89)

    if len(cfg.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(cfg.onnx_export, batch_size=1, seq_len=cfg.model.bptt)


if __name__ == '__main__':
    main()
