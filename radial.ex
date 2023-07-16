defmodule Math do
  @moduledoc """
    Módulo para funciones y ecuaciones matemáticas
  """

  @doc """
    Función de paso con umbral
  """
  def heaviside(x, bias) do
    if (x >= bias), do: 1, else: 0
  end

  @doc """
    Función gaussiana o campana de Gauss
  """
  def gaussian(x) do
    :math.exp(:math.pow(-x, 2) / 2)
  end

  @doc """
    Función que determina el valor medio en base a n de un número real
  """
  def mean(value, n) do
    value / n
  end

  @doc """
    Función de determina el error en base a un valor esperado y el valor actual
  """
  def error(expected, actual) do
    Enum.zip(expected, actual) |> Enum.map(fn {expected, actual} -> expected - actual end)
  end
end

defmodule Random do
  :rand.seed(:exsss, {0, 0, 42})

  @doc """
    Función que genera un número real entre -n y n
  """
  def rand_float(n) do
    :rand.uniform() * (2 * n) - n
  end

  @doc """
    Función que genera un vector de tamaño n de números reales entre -m y m
  """
  def generate_rand_list(n, m) do
    for _ <- 1..n, do: rand_float(m)
  end
end

defmodule Utils do
  @doc """
    Función que suma dos vectores
  """
  def add(list1, list2) do
    Enum.zip(list1, list2) |> Enum.map(fn {a, b} -> a + b end)
  end

  @doc """
    Función que determina la distancia euclidiana entre dos vectores
  """
  def euclidian_distance(list1, list2) do
    Enum.zip(list1, list2) |>
    Enum.reduce(0, fn {a, b}, acc -> :math.pow(a - b, 2) + acc end) |>
    :math.sqrt()
  end

  @doc """
    Función que determina un punto medio entre diversos puntos a un centro.
  """
  def mean_point(points, center) do
    n = length(points) + 1
    Enum.reduce(points, center, fn point, acc -> add(point, acc) end) |>
    Enum.map(fn x -> Math.mean(x, n) end)
  end
end

defmodule Parser do
  @doc """
    Función que elimina (reemplaza por caracter vacío) carácteres especiales de un String
  """
  def remove_special_char(vector, special_char) do
    Enum.map(vector, fn x -> String.replace(x, special_char, "") end)
  end

  @doc """
    Función que parsea un vector de Strings a vector de Enteros
  """
  def parse_str_list_to_int_list(vector) do
    Enum.map(vector, fn x -> String.to_integer(x) end)
  end

  @doc """
    Función que parsea un vector de Strings a vector de Flotantes (Reales)
  """
  def parse_str_list_to_float_list(vector) do
    Enum.map(vector, fn x -> String.to_float(x) end)
  end

  @doc """
    Función que parsea un vector de Flotantes (Reales) a vector de String
  """
  def parse_float_list_to_str_list(vector) do
    Enum.map(vector, fn x -> Float.to_string(x) end)
  end

  @doc """
    Función que parsea un vector de Strings a vector de Números (Enteros o Flotantes, dependiendo del modo)
  """
  def parse_str_list_to_number_list(vector, mode) do
    case mode do
     "int" -> parse_str_list_to_int_list(vector)
     "float" -> parse_str_list_to_float_list(vector)
    end
  end

  @doc """
    Función que separa un String de datos dependiendo de un patrón y, posteriormente,
    separa los strings resultantes por los caracteres de coma (,)
  """
  def split_data(data, pattern) do
    String.split(data, pattern) |> Enum.map(fn line -> String.split(line, ",") end)
  end

  @doc """
    Función que parsea archivo completo a un vector de vectores de números
  """
  def parse_file(path, mode) do
    {message, data} = File.read(path)
    case message do
       :ok ->
         split_data(data, "\n") |>
         Enum.map(fn vector -> remove_special_char(vector, "\r") end) |>
         Enum.map(fn vector -> parse_str_list_to_number_list(vector, mode) end)
       _ -> :file.format_error(data)
    end
  end

  @doc """
    Función que escribe en un archivo los datos dentro de una red neuronal
  """
  def write(path, network, type) do
    neurons = RadialNetwork.get_neurons(network, type)
    neurons_str_list = for neuron <- neurons do
     parse_float_list_to_str_list(neuron.vector) |> Enum.concat([Float.to_string(neuron.value)]) |>
     Enum.join(",")
    end
    neurons_str = Enum.join(neurons_str_list, "\n")
    File.write(path, neurons_str)
  end

  @doc """
    Función que exporta los datos de una red neuronal en dos archivos diferentes
  """
  def export(network) do
    write("base/hidden_layer.csv", network, "hidden")
    write("base/output_layer.csv", network, "output")
  end

  @doc """
    Función que importa los datos de una red neuronal que se encuentran en dos archivos diferentes
  """
  def import() do
    radial_neurons = parse_file("base/hidden_layer.csv", "float") |> Enum.map(fn hidden -> %RadialNeuron{
      centroid: Enum.slice(hidden, 0..(length(hidden) - 2)),
      desviation: Enum.at(hidden, length(hidden) - 1)}
    end)
    output_neurons = parse_file("base/output_layer.csv", "float") |> Enum.map(fn output -> %OutputNeuron{
      weights: Enum.slice(output, 0..(length(output) - 2)),
      bias: Enum.at(output, length(output) - 1)}
    end)
    %RadialNetwork{hidden: radial_neurons, output: output_neurons}
  end

  @doc """
    Función que importa los patrones de entrenamiento para una red neuronal
  """
  def get_patterns() do
    path = "training/"
    dataset = parse_file(path <> "dataset.csv", "int")
    expected = parse_file(path <> "expected.csv", "int")
    for {input, output} <- Enum.zip(dataset, expected), do: %{input: input, expected: output}
  end
end

defmodule KMeans do
  @moduledoc """
  Módulo que contiene todas las funciones necesarias para realizar el Algoritmo de K-Medios
  """

  @doc """
    Función que inicia el Algoritmo de K-Medios

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - k: Número entero que representa la cantidad de centros
  """
  def run(dataset, k, max_iterations) when is_list(dataset) do
    initial_centroids = initialize_centroids(dataset, k)
    initial_clusters = assign_clusters(dataset, initial_centroids)
    run(dataset, initial_centroids, initial_clusters, 0, max_iterations)
  end

  @doc """
    Función recursiva que calcula los nuevos centroides y los clusters por iteraciones

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - centroids: Lista de vectores de tamaño k que representan los centros actuales
      - clusters: Lista de tuplas (dato, índice) que representa un punto asociado a un centro
      - iteration: Número entero que representa la iteración actual de la función
  """
  defp run(dataset, centroids, clusters, iteration, max_iterations) do
    if (iteration < max_iterations) do
      new_centroids = recalculate_centroids(clusters, centroids)
      new_clusters = assign_clusters(dataset, new_centroids)
      run(dataset, new_centroids, new_clusters, iteration + 1, max_iterations)
    else
      %{centroids: centroids, clusters: clusters}
    end
  end

  @doc """
    Función que inicializa los centros eligiendo de manera aleatoria un punto en el dataset.

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - k: Número entero que representa la cantidad de centros
  """
  def initialize_centroids(dataset, k) do
    for _ <- 1..k, do: Enum.random(dataset)
  end

  @doc """
    Función que asigna los puntos dentro del conjunto de datos a un centro dependiendo de su distancia.
    Criterio: se elige el centro al que menor distancia tenga.

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - centroids: Lista de vectores de tamaño k que representan los centros actuales
  """
  def assign_clusters(dataset, centroids) do
    for data <- dataset do
      centroid_index = Enum.map(centroids, fn centroid -> Utils.euclidian_distance(data, centroid) end) |>
      Enum.with_index() |>
      Enum.min() |>
      elem(1)
      {data, centroid_index}
    end
  end

  @doc """
    Función que recalcula los centros a través del punto medio entre el centro y todos los
    puntos dentro del cluster

    ## Parametros
      - clusters: Lista de tuplas (dato, índice) que representa un punto asociado a un centro
      - centroids: Lista de vectores de tamaño k que representan los centros actuales
  """
  def recalculate_centroids(clusters, centroids) do
    indexed_centroids = Enum.with_index(centroids)
    for {centroid, index} <- indexed_centroids do
      Enum.filter(clusters, fn {_, centroid_index} -> centroid_index == index end) |>
      Enum.unzip() |>
      elem(0) |>
      Utils.mean_point(centroid)
    end
  end

  @doc """
    Función que determina la amplitud o desviación de una función de base radial como la media
    geométrica de la distancia del centro a sus dos vecinos más cercanos

    ## Parametros
      - centroids: Lista de vectores de tamaño k que representan los centros actuales
  """
  def calculate_desviations(centroids) do
    indexed_centroids = Enum.with_index(centroids)
    for {centroid, index} <- indexed_centroids do
      Enum.filter(indexed_centroids, fn {_, centroid_index} -> centroid_index != index end) |>
      Enum.map(fn {neighbor, _} -> Utils.euclidian_distance(centroid, neighbor) end) |>
      Enum.sort(:asc) |>
      Enum.take(2) |>
      Enum.reduce(1, fn neighbor, acc -> neighbor * acc end) |>
      :math.sqrt()
    end
  end

  @doc """
    Función que realiza el método del codo para determinar cuál es el número óptimo de centros para
    un conjunto de datos.

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - centroid_num: Número entero que representa la cantidad máxima de centros
      con los que se hará el método del codo
  """
  def elbow(dataset, centroid_num, max_iterations) do
    for k <- 1..centroid_num do
      kmean = KMeans.run(dataset, k, max_iterations)
      distorsion = distorsion(kmean)
      {k, distorsion}
    end
  end

  @doc """
    Función que determina la distorsión o varianza del Algoritmo de K-Medios a través de la
    Suma de Cuadrados Dentro del Cluster (WCSS: Within-Cluster Sum of Squares)

    ## Parametros
      - kmeans: Mapa {centroids, clusters} que representa la salida del Algoritmo de K-Medios
  """
  def distorsion(kmeans) do
    indexed_centroids = Enum.with_index(kmeans.centroids)
    cluster_distances = for {centroid, index} <- indexed_centroids do
      Enum.filter(kmeans.clusters, fn {_, centroid_index} -> centroid_index == index end) |>
      Enum.reduce(0, fn {data, _}, acc -> (Utils.euclidian_distance(data, centroid) |> :math.pow(2)) + acc end)
    end
    Enum.sum(cluster_distances)
  end
end

defmodule OutputNeuron do
  defstruct [:weights, :bias]

  @doc """
    Función que crea una neurona de capa de salida
  """
  def create(weights_num, limit_num) do
    %OutputNeuron{weights: Random.generate_rand_list(weights_num, limit_num), bias: Random.rand_float(limit_num)}
  end

  @doc """
    Función de propagación de la capa de salida (suma ponderada entre entradas y pesos)
  """
  def net_input(neuron, inputs) do
    Enum.zip(inputs, neuron.weights) |>
    Enum.map(fn {x, w} -> x * w end) |>
    Enum.reduce(0, fn x, acc -> acc + x end)
  end

  @doc """
    Función de activación de la capa de salida (función heaviside o de paso)
  """
  def activation(neuron, inputs) do
    net_input(neuron, inputs) |>
    Math.heaviside(neuron.bias)
  end

  @doc """
    Función de ajuste utilizada para el entrenamiento de la neurona que
    aumenta o disminuye los pesos y el umbral de la neurona
  """
  def adjust(neuron, learning_rate, error, inputs) do
    %{neuron | bias: adjust_bias(neuron, learning_rate, error), weights: adjust_weights(neuron, learning_rate, error, inputs)}
  end

  @doc """
    Función privada de ajuste utilizada para el entrenamiento de la neurona que
    aumenta o disminuye los pesos de la neurona
  """
  defp adjust_weights(neuron, learning_rate, error, inputs) do
    Enum.zip(inputs, neuron.weights) |> Enum.map(fn {x, w} -> w + learning_rate * error * x end)
  end

  @doc """
    Función privada de ajuste utilizada para el entrenamiento de la neurona que
    aumenta o disminuye el umbral de la neurona
  """
  defp adjust_bias(neuron, learning_rate, error) do
    neuron.bias + learning_rate * error
  end
end

defmodule RadialNeuron do
  defstruct [:centroid, :desviation]

  @doc """
    Función que crea una neurona de capa oculta
  """
  def create(centroid, desviation) do
    %RadialNeuron{centroid: centroid, desviation: desviation}
  end

  @doc """
    Función de propagación de la capa oculta (distancia entre la entrada y
    el centro de la neurona)
  """
  def net_input(neuron, inputs) do
    Utils.euclidian_distance(inputs, neuron.centroid) / neuron.desviation
  end

  @doc """
    Función de activación de la capa oculta (función gaussiana o campana de Gauss)
  """
  def activation(neuron, inputs) do
    net_input(neuron, inputs) |>
    Math.gaussian()
  end
end

defmodule RadialNetwork do
  defstruct [:epochs, :rate, :hidden, :output]

  @doc """
    Función que crea una red neuronal en base a un conjunto de centroides, número de
    neuronas en la capa de salida, número de épocas de entrenamiento, un valor de
    ratio de entrenamiento, y un límite de los pesos y umbral de la capa de salida
  """
  def create(kmeans, output_num, epochs \\ 2500, rate \\ 0.05, limit_num \\ 1) do
    desviations = KMeans.calculate_desviations(kmeans.centroids)
    hidden = for {centroid, desviation} <- Enum.zip(kmeans.centroids, desviations), do: RadialNeuron.create(centroid, desviation)
    output = for _ <- 1..output_num, do: OutputNeuron.create(length(hidden), limit_num)
    %RadialNetwork{epochs: epochs, rate: rate, hidden: hidden, output: output}
  end

  @doc """
    Función de apoyo para realizar la propagación hacia adelante en la capa oculta
  """
  defp radial_process(network, input) do
    Enum.map(network.hidden, fn hidden -> RadialNeuron.activation(hidden, input) end)
  end

  @doc """
    Función de apoyo para realizar la propagación hacia adelante en la capa de salida
  """
  defp output_process(network, input) do
    Enum.map(network.output, fn output -> OutputNeuron.activation(output, input) end)
  end

  @doc """
    Función que analiza un conjunto de entradas para dar las respuestas con
    una red neuronal entrenada
  """
  def resolve(network, inputs) do
    for input <- inputs do
      radial_outputs = radial_process(network, input)
      output_process(network, radial_outputs)
    end
  end

  @doc """
    Función de entrenamiento (aprendizaje supervisado) de la red neuronal en base
    a los patrones y épocas pasados como parámetros
  """
  def train(network, patterns, epoch) do
    if epoch <= network.epochs do
      shuffle_patterns = Enum.shuffle(patterns)
      new_network = train(network, shuffle_patterns)
      train(new_network, patterns, epoch + 1)
    else
      network
    end
  end

  @doc """
    Función de apoyo para el entrenamiento (aprendizaje supervisado) de la red neuronal
    que evalua cada patrón dentro del vector de patrones hasta que solo quede un vector
    vacío
  """
  def train(network, patterns) do
    if not Enum.empty?(patterns) do
      [pattern | remaining] = patterns
      new_network = training(network, pattern)
      train(new_network, remaining)
    else
      network
    end
  end

  @doc """
    Función que realiza la evaluación de un patrón y, si cumple con la condición,
    ajusta los pesos y el umbral de la capa de salida
  """
  defp training(network, pattern) do
    radial_outputs = radial_process(network, pattern.input)
    outputs = output_process(network, radial_outputs)
    errors = Math.error(pattern.expected, outputs)

    new_neurons = for {error, neuron} <- Enum.zip(errors, network.output) do
      if (round(error) != 0) do
        OutputNeuron.adjust(neuron, network.rate, error, radial_outputs)
      else
        neuron
      end
    end
    %{network | output: new_neurons}
  end

  @doc """
    Función que devuelve las neuronas de la capa deseada
  """
  def get_neurons(network, layer) do
    if layer == "hidden" do
      for neuron <- network.hidden, do: %{vector: neuron.centroid, value: neuron.desviation}
    else
      for neuron <- network.output, do: %{vector: neuron.weights, value: neuron.bias}
    end
  end
end

defmodule NetworkApplication do

  @doc """
    Función que devuelve las respuestas de una red neuronal importada de la
    base de conocimiento en un archivo .csv
  """
  def run() do
    inputs = Parser.parse_file("input/inputs.csv", "int")
    result_str = Parser.import() |> RadialNetwork.resolve(inputs)
    |> Enum.map(fn output -> Enum.join(output, ",") end) |> Enum.join("\n")
    File.write("output.csv", result_str)
  end

  @doc """
    Función que exporta una nueva red neuronal entrenada en dos archivos .csv
  """
  def export_new_network(k, output_num, epochs \\ 100, max_iterations \\ 120, rate \\ 0.05, limit_num \\ 1) do
    patterns = Parser.get_patterns()
    Parser.parse_file("training/dataset.csv", "int") |>
    KMeans.run(k, max_iterations) |> RadialNetwork.create(output_num, epochs, rate, limit_num) |>
    RadialNetwork.train(patterns, 0) |> Parser.export()
  end
end
