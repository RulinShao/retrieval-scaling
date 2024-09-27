import time
import random


class DatastoreAPI():
    def __init__(self, datastore_name) -> None:
        self.datastore_name = datastore_name
        self.datastore = None
    
    def search(self, item):
        # TODO: do actual search
        # time.sleep(random.random() * 2)
        item.searched_results = ["Dummpy test case."]
        print(item)
        return item

def get_datastore(datastore_name):
    return DatastoreAPI(datastore_name)