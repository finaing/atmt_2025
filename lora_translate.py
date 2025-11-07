from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq import models, utils
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq.decode import decode
import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from tqdm import tqdm

from peft import (
    LoraConfig,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int,
                        help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', required=True,
                        help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument(
        '--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument(
        '--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True,
                        help='path to the base model checkpoint file')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int,
                        help='maximum length of generated sequence')

    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true',
                        help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str,
                        help='Path to the reference file (one sentence per line, required if --bleu is set)')

    # LoRA arguments
    parser.add_argument('--lora', action='store_true',
                        help='Enable LoRA adapter at inference')
    parser.add_argument('--lora-r', default=8, type=int, help='LoRA rank (r)')
    parser.add_argument('--lora-alpha', default=32,
                        type=int, help='LoRA alpha (scaling)')
    parser.add_argument('--lora-dropout', default=0.0,
                        type=float, help='LoRA dropout')
    parser.add_argument('--lora-target-modules', default=None, type=str,
                        help='Comma-separated list of target submodules for LoRA (e.g. "q_proj,k_proj,v_proj,out_proj")')
    parser.add_argument('--lora-state-dict', default=None, type=str,
                        help='Path to LoRA adapter state dict (e.g. checkpoints/lora_best.pt). If not provided, will try to load from checkpoint.')

    return parser.parse_args()


def setup_lora(args, model):
    """Inject LoRA adapters into the model."""
    if args.lora_target_modules:
        target_modules = [
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        ]
    else:
        # Default targets for Transformer
        target_modules = ["q_proj", "k_proj",
                          "v_proj", "out_proj", "fc1", "fc2"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM",
    )

    model = inject_adapter_in_model(lora_config, model)
    logging.info(
        f"Injected LoRA adapters into model (targets: {target_modules})")
    return model


def load_checkpoint_and_lora(args, model, device):
    """
    Load model checkpoint and optionally LoRA adapters.

    Two modes:
    1. If --lora-state-dict is provided: Load base checkpoint + separate LoRA adapters
    2. Otherwise: Assume checkpoint contains both base and LoRA weights (from training)
    """
    # Load the base checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        logging.info(
            f"Loaded checkpoint from epoch {checkpoint.get('last_epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    # Check if checkpoint contains LoRA keys
    has_lora_in_checkpoint = any('lora' in key.lower()
                                 for key in state_dict.keys())

    if args.lora:
        # Inject LoRA adapters into model architecture
        model = setup_lora(args, model)

        if args.lora_state_dict:
            # Mode 1: Load base weights, then load separate LoRA adapters
            logging.info(
                "Loading base model weights and separate LoRA adapters...")

            # Load base model weights (strict=False to ignore missing LoRA keys)
            missing, unexpected = model.load_state_dict(
                state_dict, strict=False)

            # Filter out LoRA-related missing keys (expected)
            non_lora_missing = [k for k in missing if 'lora' not in k.lower()]
            if non_lora_missing:
                logging.warning(
                    f"Missing non-LoRA keys: {non_lora_missing[:5]}")

            # Now load the LoRA adapter weights
            adapter_state = torch.load(
                args.lora_state_dict, map_location=device)
            set_peft_model_state_dict(model, adapter_state)
            logging.info(f"Loaded LoRA adapters from {args.lora_state_dict}")

        else:
            # Mode 2: Load checkpoint that contains both base and LoRA weights
            if has_lora_in_checkpoint:
                logging.info(
                    "Loading checkpoint with embedded LoRA weights...")
                missing, unexpected = model.load_state_dict(
                    state_dict, strict=False)

                if missing:
                    logging.warning(
                        f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
                if unexpected:
                    logging.warning(
                        f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
            else:
                logging.warning(
                    "LoRA enabled but checkpoint doesn't contain LoRA weights. "
                    "Model will use randomly initialized LoRA adapters!"
                )
                model.load_state_dict(state_dict, strict=False)
    else:
        # No LoRA: just load base model
        logging.info("Loading base model without LoRA...")
        if has_lora_in_checkpoint:
            logging.warning(
                "Checkpoint contains LoRA weights but --lora flag not set. "
                "LoRA weights will be ignored."
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logging.warning(f"Unexpected keys: {len(unexpected)}")

    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,} | Trainable: {trainable:,}")


def postprocess_ids(ids, pad, bos, eos):
    """Remove leading BOS, truncate at first EOS, remove PADs."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # Remove leading BOS if present
    if len(ids) > 0 and ids[0] == bos:
        ids = ids[1:]
    # Truncate at EOS (do not include EOS)
    if eos in ids:
        ids = ids[:ids.index(eos)]
    # Remove PAD tokens
    ids = [i for i in ids if i != pad]
    return ids


def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids, pad, bos, eos):
    """Convert token ids to a detokenized string using the target tokenizer."""
    ids = postprocess_ids(sentence_ids, pad, bos, eos)
    return tokenizer.Decode(ids)


def batch_iter(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]


def main(args):
    """ Main translation function """
    torch.manual_seed(args.seed)

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Load tokenizers
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    # Load checkpoint to get model args
    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=lambda s, l: default_restore_location(s, 'cpu'),
        weights_only=False
    )

    # Merge checkpoint args with command line args (command line takes precedence)
    if 'args' in checkpoint:
        args_loaded = argparse.Namespace(
            **{**vars(checkpoint['args']), **vars(args)}
        )
        args = args_loaded

    # Build model architecture
    logging.info("Building model architecture...")
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)

    device = torch.device('cuda' if args.cuda else 'cpu')
    model = model.to(device)

    # Load checkpoint and LoRA adapters
    load_checkpoint_and_lora(args, model, device)

    # Set model to evaluation mode
    model.eval()

    # Read input sentences
    logging.info(f"Reading input from {args.input}")
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(src_lines)} sentences to translate")

    # Encode input sentences
    src_encoded = [
        torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True))
        for line in src_lines
    ]

    # Trim to max_len
    max_seq_len = min(model.encoder.pos_embed.size(1), args.max_len)
    src_encoded = [
        s if len(s) <= max_seq_len else s[:max_seq_len]
        for s in src_encoded
    ]

    # Get special token IDs
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()

    logging.info(f'PAD ID: {PAD}, BOS ID: {BOS}, EOS ID: {EOS}')
    logging.info(
        f'PAD: "{src_tokenizer.IdToPiece(PAD)}", BOS: "{tgt_tokenizer.IdToPiece(BOS)}", EOS: "{tgt_tokenizer.IdToPiece(EOS)}"')

    # Clear output file
    if args.output is not None:
        with open(args.output, 'w', encoding="utf-8") as out_file:
            out_file.write('')

    make_batch = utils.make_batch_input(
        device=device, pad=src_tokenizer.pad_id(), max_seq_len=max_seq_len
    )

    translations = []
    start_time = time.perf_counter()

    # Translation loop (batched)
    logging.info("Starting translation...")
    for batch in tqdm(batch_iter(src_encoded, args.batch_size),
                      total=(len(src_encoded) + args.batch_size -
                             1) // args.batch_size,
                      desc="Translating"):
        with torch.no_grad():
            # Pad the batch to the same length
            batch_lengths = [len(x) for x in batch]
            max_len = max(batch_lengths)
            batch_padded = [
                torch.cat(
                    [x, torch.full((max_len - len(x),), PAD, dtype=torch.long)])
                if len(x) < max_len else x
                for x in batch
            ]
            src_tokens = torch.stack(batch_padded).to(device)

            # Create a dummy target tensor (all PADs, same shape as src_tokens)
            dummy_y = torch.full_like(src_tokens, fill_value=PAD)

            # Use make_batch to get masks
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(
                src_tokens, dummy_y
            )

            # Decode without teacher forcing
            prediction = decode(
                model=model,
                src_tokens=src_tokens,
                src_pad_mask=src_pad_mask,
                max_out_len=args.max_len,
                tgt_tokenizer=tgt_tokenizer,
                args=args,
                device=device
            )

        # Decode each sentence and save
        for sent in prediction:
            translation = decode_sentence(tgt_tokenizer, sent, PAD, BOS, EOS)
            translations.append(translation)
            if args.output is not None:
                with open(args.output, 'a', encoding="utf-8") as out_file:
                    out_file.write(translation + '\n')

    end_time = time.perf_counter()
    logging.info(f'Wrote {len(translations)} lines to {args.output}')
    logging.info(
        f'Translation completed in {end_time - start_time:.2f} seconds')

    # Compute BLEU score if requested
    if getattr(args, 'bleu', False):
        if not args.reference:
            raise ValueError("You must provide --reference when using --bleu.")

        logging.info(f"Computing BLEU score against {args.reference}")
        with open(args.reference, encoding='utf-8') as ref_file:
            references = [line.strip() for line in ref_file if line.strip()]

        if len(references) != len(translations):
            raise ValueError(
                f"Reference ({len(references)}) and hypothesis ({len(translations)}) "
                f"line counts do not match."
            )

        bleu = sacrebleu.corpus_bleu(translations, [references])
        logging.info(f"BLEU score: {bleu.score:.2f}")
        print(f"\nBLEU score: {bleu.score:.2f}")


if __name__ == '__main__':
    args = get_args()
    main(args)
