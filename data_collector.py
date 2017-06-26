import os
import config

import fetch_data.data_client as client


def should_download_images(configured_path, required_count):
    files_count = len([name for name in os.listdir(configured_path)
                       if os.path.isfile(os.path.join(configured_path, name))])
    
    print("Number of existing images is {}.".format(files_count))
    return (not os.path.exists(configured_path) or 
            files_count < required_count)


if __name__ == '__main__':
    if should_download_images(config.SEGMENTED_LUNGS_DIR, 
                              config.REQUIRED_IMGS_COUNT):
        client.collect_images(config.BUCKET_IN_USE, 
                              config.PROJECT_NAME, 
                              config.FETCHED_DATA_DIR)
    else:
        print("Images required for trainig the selected",
              "model have already been downloaded to: ",
               config.SEGMENTED_LUNGS_DIR)