from .soccer_model import SoccerModel
def create_model(opt):
    model = None
    model = SoccerModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
