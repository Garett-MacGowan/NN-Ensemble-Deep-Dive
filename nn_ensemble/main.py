if __name__ == '__main__':
    import sys
    from nn_ensemble.controller import Controller
    from nn_ensemble.configurations import configuration

    controller: Controller = Controller()
    controller.execute(sys.argv[1:], configuration.get())
