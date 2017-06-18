import config

import fetch_data.data_client as client


if __name__ == '__main__':
    if config.download_images:
        client.collect_images(config.BUCKET_IN_USE, 
                              config.PROJECT_NAME, 
                              config.FETCHED_DATA_DIR)
    else:
        print("Images required for trainig the selected",
              "model have already been downloaded to: ",
              config.SEGMENTED_LUNGS_DIR)