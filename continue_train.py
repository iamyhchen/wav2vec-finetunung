import argparse
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Dict, List, Union
from jiwer import wer
import numpy as np

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer(label_str, pred_str)}

def main():
    parser = argparse.ArgumentParser(description="Continue Wav2Vec2 Training from Checkpoint")
    parser.add_argument('--train_csv', type=str, default='dataset/train/train.csv')
    parser.add_argument('--val_csv', type=str, default='dataset/val/val.csv')
    parser.add_argument('--output_dir', type=str, default='./wav2vec2-finetuned-opdir')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_strategy', type=str, default='steps')
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--pretrained_model', type=str, default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint directory')
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("csv", data_files={
        "train": args.train_csv,
        "validation": args.val_csv
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Tokenizer & Processor
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=args.vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
        batch["attention_mask"] = processor(audio["array"], sampling_rate=16000).attention_mask[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

    # Model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none"
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )

    # 從 checkpoint 繼續訓練
    trainer.train(resume_from_checkpoint=args.checkpoint_path)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
