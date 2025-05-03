import tensorflow as tf
import numpy as np

tf.constant(u'Thanks 😊')

tf.constant([u"You're", u"welcome!"]).shape

# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8= tf.constant(u"语言处理")
print(text_utf8)


# Unicode string, represented as a vector of Unicode code points.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print(text_chars)

tf.strings.unicode_decode(text_utf8, input_encoding='UTF_8')
tf.strings.unicode_encode(text_chars,
                          output_encoding='UTF-8')
tf.strings.unicode_transcode(text_utf8,
                             input_encoding='UTF8',
                             output_encoding='UTF-16-BE')

print(tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [99, 111, 119]],
                                output_encoding='UTF-8'))

# A batch of Unicode strings, each represented as a UTF8-encoded string.
batch_utf8 = [s.encode('UTF-8') for s in
              [u'hÃllo', u'What is the weather tomorrow', u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8,
                                               input_encoding='UTF-8')
batch_chars_sparse = batch_chars_ragged.to_sparse()
#for different length string:
tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse),
    output_encoding='UTF-8')

thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))
print(tf.strings.unicode_split(thanks, 'UTF-8').numpy())
