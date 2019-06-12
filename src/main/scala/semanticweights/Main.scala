package semanticweights

import breeze.linalg.DenseMatrix
import io.picnicml.doddlemodel.data.Features
import org.tensorflow.{SavedModelBundle, Tensor}

object Main extends App {
  val model = SavedModelBundle.load("universal_sentence_encoder_large_v3", "serve")
  val embeddingsDimension = 512

  val sentences = Array("hello world sentence", "this works!", "we should get three embeddings")
  val embeddings = embed(sentences)

  model.session().close()

  println(s"shape of the embeddings matrix: (${embeddings.rows}, ${embeddings.cols})")
  println(s"preview of the embeddings matrix:\n$embeddings")

  def embed(sentences: Array[String]): Features = {
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

  private def getEmbeddings(computationResult: Tensor[_]): Features = {
    val embeddings = computationResult
      .copyTo(sentences.map(_ => Array.fill[Float](embeddingsDimension) { 0 }))
      .map(_.map(_.toDouble))
    DenseMatrix(embeddings:_*)
  }
}
