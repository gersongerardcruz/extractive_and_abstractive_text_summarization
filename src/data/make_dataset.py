import click
import logging
from pathlib import Path
from preprocessing import DataProcessor

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    preprocessor = DataProcessor()

    logger.info('reading data')
    preprocessor.read_data(input_filepath, 0)

    logger.info('processing data')
    preprocessor.process_data()

    logger.info('saving processed data')
    preprocessor.write_data(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
