


class DatastoreAPI():
    def __init__(self, datastore_name) -> None:
        self.datastore_name = datastore_name
        self.datastore = None
    
    def search(self, item):
        # TODO: do actual search
        item.searched_results = ["Dummpy test case."]

        return item

def get_datastore(datastore_name):
    raise DatastoreAPI(datastore_name)