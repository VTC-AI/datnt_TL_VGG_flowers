from keras.applications import VGG16
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model


base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
# x = GlobalAveragePooling2D()(x)
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

preds = Dense(17, activation='softmax')(x)

model = model = Model(inputs=base_model.input, outputs=preds)

for layer in base_model.layers[:15]:
  layer.trainable = False
for layer in base_model.layers[15:]:
  layer.trainable = True

opt = RMSprop(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
