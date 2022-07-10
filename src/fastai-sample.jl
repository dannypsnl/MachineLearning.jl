using FastAI

data, blocks = load(datarecipes()["imagenette2-320"])
task = ImageClassificationSingle(blocks, size=(256, 256))
learner = tasklearner(task, data, callbacks=[ToGPU(), Metrics(accuracy)])
fitonecycle!(learner, 1, 0.1)
showoutputs(task, learner)