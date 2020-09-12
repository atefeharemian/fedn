import os
import threading

from fedn.clients.reducer.control import ReducerControl
from fedn.clients.reducer.interfaces import ReducerInferenceInterface
from fedn.clients.reducer.restservice import ReducerRestService
from fedn.clients.reducer.state import ReducerStateToString
from fedn.common.security.certificatemanager import CertificateManager


class Reducer:
    def __init__(self, config):
        self.name = config['name']
        self.token = config['token']

        try:
            path = config['path']
        except KeyError:
            path = os.getcwd()

        self.certificate_manager = CertificateManager(os.getcwd() + "/certs/")

        self.control = ReducerControl()
        self.inference = ReducerInferenceInterface()
        rest_certificate = self.certificate_manager.create("reducer")
        self.rest = ReducerRestService(config['name'], self.control, rest_certificate)

    def run(self):

        threading.Thread(target=self.rest.run, daemon=True).start()

        import time
        try:
            while True:
                time.sleep(1)
                print("Reducer in {} state".format(ReducerStateToString(self.control.state())), flush=True)
                self.control.monitor()
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)
