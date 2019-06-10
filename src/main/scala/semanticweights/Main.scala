package semanticweights

import org.platanios.tensorflow.api.tensors.Tensor

object Main extends App {
  val t1 = Tensor(1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  val t3 = t1 + t2
  println(t3.summarize())
}
