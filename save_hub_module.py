import tensorflow as tf
import tensorflow_hub as hub

graph = tf.Graph()
with graph.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None], name='text_input')
    sentence_encoder = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3', trainable=True)
    embedded_text = sentence_encoder(text_input)
    embedded_text = tf.identity(embedded_text, name='embedded_text')
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

with tf.Session(graph=graph) as session:
    session.run(init_op)
    tf.saved_model.simple_save(
        session,
        'universal_sentence_encoder_large_v3',
        inputs={'text_input': text_input},
        outputs={'embedded_text': embedded_text},
        legacy_init_op=init_op,
    )

graph.finalize()
