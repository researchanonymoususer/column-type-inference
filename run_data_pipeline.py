import argparse
import csv
import logging
import os

from gensim.models import KeyedVectors

from utils.data_pipeline import DataPipeline
from utils.backoff_embedding_generator import BackedOffEmbedding

# --- Constants ---
EMBEDDING_MODEL_PATH = 'model/crawl-300d-2M-subword.vec'
DESTINATION_FILE = 'data/features_auto_labeled.csv'
OOV_RESULTS_FILE = 'data/oov_results_fast_text.csv'
LOG_FILE = 'logs/data_pipeline.log'
DEFAULT_BATCH_SIZE = 1000
DEFAULT_BATCH_NUMBER = 1


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the data pipeline on a batch of source files.')
    parser.add_argument('file_path', type=str, help='Path to the file containing source file paths')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Number of files per batch')
    parser.add_argument('--batch-number', type=int, default=DEFAULT_BATCH_NUMBER, help='Batch number to process (1-indexed)')
    return parser.parse_args()


def load_source_files(file_path: str) -> list[str]:
    with open(file_path, 'r') as f:
        return [line.rstrip('\n\r') for line in f]


def save_oov_results(oov_results: list[dict], output_file: str) -> None:
    file_exists = os.path.exists(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['OOV', 'Best_Candidates'])
        if not file_exists:
            writer.writeheader()
        writer.writerows(oov_results)


def run_pipeline(batch: list[str], embedder: BackedOffEmbedding, logger: logging.Logger) -> tuple[int, int]:
    num_files = 0
    num_columns_processed = 0

    for source_file in batch:
        num_files += 1
        sep = '=' * 80
        logger.info(f"\n{sep}\nProcessing file {num_files}: {source_file}\n{sep}")

        try:
            pipeline = DataPipeline(source_file, DESTINATION_FILE, logger, embedder)
            pipeline.run_pipeline()
            num_columns_processed += len(pipeline.features)
        except Exception as e:
            logger.error(f"Failed to process {source_file}: {e}")

    return num_files, num_columns_processed


def main() -> None:
    args = parse_args()
    logger = setup_logging(LOG_FILE)

    logger.info("Loading embedding model...")
    embedding_model = KeyedVectors.load_word2vec_format(EMBEDDING_MODEL_PATH, binary=False)
    embedder = BackedOffEmbedding(embedding_model, logger, ngram_range=(3, 5))

    source_files = load_source_files(args.file_path)

    start_idx = (args.batch_number - 1) * args.batch_size
    end_idx = args.batch_number * args.batch_size
    batch = source_files[start_idx:end_idx]

    logger.info(f"Processing batch {args.batch_number} (files {start_idx + 1}–{end_idx}): {len(batch)} files")

    num_files, num_columns_processed = run_pipeline(batch, embedder, logger)

    sep = '=' * 80
    logger.info(f"\n{sep}\nPIPELINE COMPLETE\n{sep}")
    logger.info(f"Total files processed:   {num_files}")
    logger.info(f"Total columns processed: {num_columns_processed}")
    logger.info(f"Output saved to:         {DESTINATION_FILE}")

    save_oov_results(embedder.oov_results, OOV_RESULTS_FILE)


if __name__ == "__main__":
    main()
