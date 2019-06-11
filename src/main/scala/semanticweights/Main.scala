package semanticweights

import org.tensorflow.{SavedModelBundle, Tensor}


object Main extends App {
  val model = SavedModelBundle.load("universal_sentence_encoder_large_v3", "serve")

  val sentences = Array("hello world sentence", "this works!")
  val input = prepareInput(sentences)

  val computationResult = model
    .session()
    .runner()
    .feed("text_input", input)
    .fetch("embedded_text")
    .run()
    .get(0)

  println(computationResult)
  println(getEmbeddings(computationResult))

  def prepareInput(sentences: Array[String]): Tensor[String] =
    Tensor.create(sentences.map(_.getBytes("UTF-8")), classOf[String])

  def getEmbeddings(computationResult: Tensor[_]): List[List[Double]] = {
    val embeddings = computationResult.copyTo(sentences.map(_ => Array.fill[Float](512) { 0 }))
    embeddings.map(_.map(_.toDouble).toList).toList
  }
}
