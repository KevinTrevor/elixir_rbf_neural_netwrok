defmodule Math do
  def heaviside(x, bias) do
    if (x >= bias), do: 1, else: 0
  end

  def gaussian(x) do
    :math.exp(:math.pow(-x, 2) / 2)
  end

  def mean(value, n) do
    value / n
  end

  def error(expected, actual) do
    Enum.zip(expected, actual) |> Enum.map(fn {expected, actual} -> expected - actual end)
  end
end

defmodule Random do
  :rand.seed(:exsss, {0, 0, 42})

  def rand_float() do
    :rand.uniform() * 2 - 1
  end

  def rand_int(n) do
    :rand.uniform(n - 1)
  end

  def generate_rand_list(n) do
    for _ <- 1..n, do: rand_float()
  end
end

defmodule Utils do
  def add(list1, list2) do
    Enum.zip(list1, list2) |> Enum.map(fn {a, b} -> a + b end)
  end

  def euclidian_distance(list1, list2) do
    Enum.zip(list1, list2) |>
    Enum.reduce(0, fn {a, b}, acc -> :math.pow(a - b, 2) + acc end) |>
    :math.sqrt()
  end

  def mean_point(points, center) do
    n = length(points) + 1
    Enum.reduce(points, center, fn point, acc -> add(point, acc) end) |>
    Enum.map(fn x -> Math.mean(x, n) end)
  end
end

defmodule Parser do
  def remove_special_char(vector, special_char) do
    Enum.map(vector, fn x -> String.replace(x, special_char, "") end)
  end

  def parse_str_list_to_int_list(vector) do
    Enum.map(vector, fn x -> String.to_integer(x) end)
  end

  def parse_str_list_to_float_list(vector) do
    Enum.map(vector, fn x -> String.to_float(x) end)
  end

  def parse_float_list_to_str_list(vector) do
    Enum.map(vector, fn x -> Float.to_string(x) end)
  end

  def parse_list_to_str_list(vector, mode) do
    case mode do
     "int" -> parse_str_list_to_int_list(vector)
     "float" -> parse_str_list_to_float_list(vector)
    end
  end

  def split_data(data, pattern) do
    String.split(data, pattern) |> Enum.map(fn line -> String.split(line, ",") end)
  end

  def parse_file(path, mode) do
    {message, data} = File.read(path)
    case message do
       :ok ->
         split_data(data, "\n") |>
         Enum.map(fn vector -> remove_special_char(vector, "\r") end) |>
         Enum.map(fn vector -> parse_list_to_str_list(vector, mode) end)
       _ -> :file.format_error(data)
    end
  end

  def write_hidden(path, network) do
    neurons_str_list = for neuron <- network.hidden do
     information = parse_float_list_to_str_list(neuron.centroid) |> Enum.concat([Float.to_string(neuron.desviation)])
     Enum.join(information, ",")
    end
    neurons_str = Enum.join(neurons_str_list, "\n")
    File.write(path, neurons_str)
  end

  def write_output(path, network) do
    neurons_str_list = for neuron <- network.output do
     information = parse_float_list_to_str_list(neuron.weights) |> Enum.concat([Float.to_string(neuron.bias)])
     Enum.join(information, ",")
    end
    neurons_str = Enum.join(neurons_str_list, "\n")
    File.write(path, neurons_str)
  end

  def export(network) do
    write_hidden("base/hidden.csv", network)
    write_output("base/output.csv", network)
  end

  def import() do
    hidden_list = parse_file("base/hidden.csv", "float")
    output_list = parse_file("base/output.csv", "float")
    radial_neurons = for hidden <- hidden_list do
      centroid = Enum.slice(hidden, 0..(length(hidden) - 2))
      desviation = Enum.at(hidden, length(hidden) - 1)
      %RadialNeuron{centroid: centroid, desviation: desviation}
    end
    output_neurons = for output <- output_list do
      weights = Enum.slice(output, 0..(length(output) - 2))
      bias = Enum.at(output, length(output) - 1)
      %OutputNeuron{weights: weights, bias: bias}
    end
    %RadialNetwork{hidden: radial_neurons, output: output_neurons}
  end

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
  def run(dataset, k) when is_list(dataset) do
    initial_centroids = initialize_centroids(dataset, k)
    initial_clusters = assign_clusters(dataset, initial_centroids)
    run(dataset, initial_centroids, initial_clusters, 0)
  end

  @doc """
    Función recursiva que calcula los nuevos centroides y los clusters por iteraciones

    ## Parametros
      - dataset: Lista de vectores que representan puntos en el plano o espacio
      - centroids: Lista de vectores de tamaño k que representan los centros actuales
      - clusters: Lista de tuplas (dato, índice) que representa un punto asociado a un centro
      - iteration: Número entero que representa la iteración actual de la función
  """
  defp run(dataset, centroids, clusters, iteration) do
    if (iteration < 10) do
      new_centroids = recalculate_centroids(clusters, centroids)
      new_clusters = assign_clusters(dataset, new_centroids)
      run(dataset, new_centroids, new_clusters, iteration + 1)
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
  def elbow(dataset, centroid_num) do
    for k <- 1..centroid_num do
      kmean = KMeans.run(dataset, k)
      distorsion = distorsion(kmean)
      {kmean, distorsion}
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
      Enum.reduce(0, fn {data, _}, acc -> (Utils.euclidian_distance(centroid, data) |> :math.pow(2)) + acc end)
    end
    Enum.sum(cluster_distances)
  end
end

defmodule OutputNeuron do
  defstruct [:weights, :bias]

  def create(weights_num) do
    %OutputNeuron{weights: Random.generate_rand_list(weights_num), bias: Random.rand_float()}
  end

  def net_input(neuron, inputs) do
    Enum.zip(inputs, neuron.weights) |>
    Enum.map(fn {x, w} -> x * w end) |>
    Enum.reduce(0, fn x, acc -> acc + x end)
  end

  def activation(neuron, inputs) do
    net_input(neuron, inputs) |>
    Math.heaviside(neuron.bias)
  end

  def adjust(neuron, learning_rate, error, inputs) do
    %{neuron | bias: adjust_bias(neuron, learning_rate, error), weights: adjust_weights(neuron, learning_rate, error, inputs)}
  end

  defp adjust_weights(neuron, learning_rate, error, inputs) do
    Enum.zip(inputs, neuron.weights) |> Enum.map(fn {x, w} -> w + learning_rate * error * x end)
  end

  defp adjust_bias(neuron, learning_rate, error) do
    neuron.bias + learning_rate * error
  end
end

defmodule RadialNeuron do
  defstruct [:centroid, :desviation]

  def create(centroid, desviation) do
    %RadialNeuron{centroid: centroid, desviation: desviation}
  end

  def net_input(neuron, inputs) do
    Utils.euclidian_distance(inputs, neuron.centroid) / neuron.desviation
  end

  def activation(neuron, inputs) do
    net_input(neuron, inputs) |>
    Math.gaussian()
  end

  def competition(outputs) do
    result = for _ <- 1..length(outputs), do: 0
    index = Enum.find_index(outputs, fn x -> x == Enum.min(outputs) end)
    List.replace_at(result, index, Enum.min(outputs))
  end
end

defmodule RadialNetwork do
  defstruct [:iterations, :rate, :hidden, :output]

  def create(kmeans, output_num, iterations \\ 2500, rate \\ 0.05) do
    desviations = KMeans.calculate_desviations(kmeans.centroids)
    hidden = for {centroid, desviation} <- Enum.zip(kmeans.centroids, desviations), do: RadialNeuron.create(centroid, desviation)
    output = for _ <- 1..output_num, do: OutputNeuron.create(length(hidden))
    %RadialNetwork{iterations: iterations, rate: rate, hidden: hidden, output: output}
  end

  defp radial_process(network, input) do
    for neuron <- network.hidden, do: RadialNeuron.activation(neuron, input)
  end

  defp output_process(network, input) do
    for neuron <- network.output, do: OutputNeuron.activation(neuron, input)
  end

  def resolve(network, inputs) do
    for input <- inputs do
      radial_outputs = radial_process(network, input) |> RadialNeuron.competition()
      output_process(network, radial_outputs)
    end
  end

  def train(network, patterns) do
    train(network, patterns, 1)
  end

  defp train(network, patterns, iteration) do
    if iteration <= network.iterations do
      new_network = training(network, patterns)
      train(new_network, patterns, iteration + 1)
    else
      network
    end
  end

  defp training(network, patterns) do
    pattern = Enum.random(patterns)
    radial_outputs = radial_process(network, pattern.input) |> RadialNeuron.competition()
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
end

defmodule NetworkApplication do
  def run() do
    nn = Parser.import()
    inputs = Parser.parse_file("input/inputs.csv", "int")
    RadialNetwork.resolve(nn, inputs)
  end

  def export_new_network(k, output_num) do
    dataset = Parser.parse_file("training/dataset.csv", "int")
    kmeans = KMeans.run(dataset, k)
    nn = RadialNetwork.create(kmeans, output_num)
    patterns = Parser.get_patterns()
    trained_nn = RadialNetwork.train(nn, patterns)
    Parser.export(trained_nn)
  end
end
