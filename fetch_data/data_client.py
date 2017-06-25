import os

from google.cloud import storage
import googleapiclient.discovery

from oauth2client.client import GoogleCredentials


credentials = GoogleCredentials.get_application_default()


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


def collect_images(bucket_name, project_name, working_dir='./'):
    all_blobs = map(lambda item: item['name'], list_bucket(bucket_name))

    client = storage.Client(project=project_name)
    bucket = client.get_bucket(bucket_name)

    for blob_item in all_blobs:
        blob = storage.Blob(blob_item, bucket)
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