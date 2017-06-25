import os
import multiprocessing
import concurrent.futures
import numpy as np

from google.cloud import storage
from google.cloud.storage import Blob

import googleapiclient.discovery

from oauth2client.client import GoogleCredentials


credentials = GoogleCredentials.get_application_default()
NUM_PROCESSES = multiprocessing.cpu_count()


def create_service():
    # Construct the service object for interacting with the Cloud Storage API -
    # the 'storage' service, at version 'v1'.
    # You can browse other available api services and versions here:
    #     http://g.co/dv/api-client-library/python/apis/
    return googleapiclient.discovery.build('storage', 'v1', credentials=credentials)


def list_bucket(bucket):
    """Returns a list of metadata of the objects within the given bucket."""
    service = create_service()

    # Create a request to objects.list to retrieve a list of objects.
    fields_to_return = \
        'nextPageToken,items(name,size,contentType,metadata(my-key))'
    req = service.objects().list(bucket=bucket, fields=fields_to_return)

    all_objects = []
    # If too many items to list in one request, list_next() will
    # automatically handle paging with the pageToken.
    while req:
        resp = req.execute()
        all_objects.extend(resp.get('items', []))
        req = service.objects().list_next(req, resp)

    return all_objects


def collect_blobs_chunk(blobs, project_name, bucket_name, working_dir):
    client = storage.Client(project=project_name)
    bucket = client.get_bucket(bucket_name)

    for blob_item in blobs:
        blob = Blob(blob_item, bucket)
        complete_path = os.path.join(working_dir, blob_item)
        dir_name = os.path.dirname(complete_path)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(complete_path, 'wb') as file_obj:
            try:
                blob.download_to_file(file_obj)
                print("Stored blob path: ", complete_path)
            except Exception as e:
                print("Downloading {} failed with {}.".format(complete_path, e))


def collect_images(bucket_name, project_name, working_dir='./'):
    all_blobs = [item['name'] for item in list_bucket(bucket_name)]
    
    chunked_data = np.array_split(all_blobs, NUM_PROCESSES)
    print("Number of processes: {}, total chunks of data {}!".format(
        NUM_PROCESSES, len(chunked_data)))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PROCESSES)

    futures = []
    for i, blob_chunk in enumerate(chunked_data):
        try:
            f = executor.submit(collect_blobs_chunk,
                                blob_chunk,
                                project_name,
                                bucket_name, 
                                working_dir)
            print("Submit {} batch to executor!".format(i))
            futures.append(f)
        except Exception as e:
            print("An error occured while downloading blobs chunk in feature {}: {}".format(
                str(i), e))

    print(concurrent.futures.wait(futures)) # By defaults waits for all
    print("Shutdown and wait for processes!")
    executor.shutdown(wait=True)