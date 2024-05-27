import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Datasets/SMSSpamCollection.txt', sep='\t', names=["label", "message"])
print(df.head())

X = list(df['message'])
y = list(df['label'])

y = list(pd.get_dummies(y, drop_first=True)['spam'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(100).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.fit(train_dataset, epochs=2, validation_data=test_dataset)

    eval_loss, eval_accuracy = model.evaluate(test_dataset)
    print(f"Evaluation loss: {eval_loss}")
    print(f"Evaluation accuracy: {eval_accuracy}")

    predictions = model.predict(test_dataset).logits
    predicted_labels = tf.argmax(predictions, axis=1).numpy()

cm = confusion_matrix(y_test, predicted_labels)
print(cm)

model.save_pretrained('senti_model')