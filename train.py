import dataset
from model import model


# step_size_train = dataset.train_generator.n // dataset.train_generator.batch_size
step_size_train = len(dataset.X_train) // 32
# step_size_test = len(dataset.X_test) // dataset.test_generator.batch_size
epoch = 20

model.fit_generator(
  generator=dataset.train_generator,
  steps_per_epoch=step_size_train,
  # validation_data=dataset.test_generator,
  # validation_steps=step_size_test,
  epochs=epoch
)

model.save('model.h5')
