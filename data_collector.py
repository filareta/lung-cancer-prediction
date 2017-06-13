import config

import fetch_data.data_client as client


if __name__ == '__main__':
    client.collect_images(config.BUCKET_IN_USE, 
                          config.PROJECT_NAME, 
                          config.FETCHED_DATA_DIR)