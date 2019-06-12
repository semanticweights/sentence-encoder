package semanticweights

import org.tensorflow.{SavedModelBundle, Tensor}

object Main extends App {
  val model = SavedModelBundle.load("universal_sentence_encoder_large_v3", "serve")
  val sentences = Array("hello world sentence", "this works!")
  val embeddings = embed(sentences)
  model.session().close()
  println(s"number of embeddings: ${embeddings.length}")
  println(embeddings)

  def embed(sentences: Array[String]): List[List[Double]] = {
    val computationResult = model
      .session()
      .runner()
      .feed("text_input", prepareInput(sentences))
      .fetch("embedded_text")
      .run()
      .get(0)
    getEmbeddings(computationResult)
  }

  private def prepareInput(sentences: Array[String]): Tensor[String] =
    Tensor.create(sentences.map(_.getBytes("UTF-8")), classOf[String])

  private def getEmbeddings(computationResult: Tensor[_]): List[List[Double]] = {
    computationResult
      .copyTo(sentences.map(_ => Array.fill[Float](512) { 0 }))
      .map(_.map(_.toDouble).toList)
      .toList
  }
}
