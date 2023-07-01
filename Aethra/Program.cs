// Carry me off to the sky...

using System.Diagnostics;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

TicTacToe game = new TicTacToe(false, false);
game.PlayGame(100000, 100);
game.Player1Human = true;
game.PlayGame(5, 1);
 
void NetworkTest(int inputs, int outputs, List<int> hiddenLayerSizes)
{
    Random random = new Random();
    NeuralNet testNeuralNet = new NeuralNet(inputs, outputs, hiddenLayerSizes, random);
    List<double> output = testNeuralNet.Ask(new List<double>() { 1, 1, 1 });
    Console.WriteLine(string.Join(", ", output));
    List<double> expected = new List<double>() { 1, 0 };
    Gradient gradient = new Gradient(testNeuralNet, new List<double>() { 1, 1, 1 }, expected);
    gradient.ApplyGradient(testNeuralNet, 1);

}

void AdditionTest(int cycles)
{
    Random random = new Random();
    NeuralNet testNeuralNet = new NeuralNet(2, 1, new List<int>() , random);
    double cost = 0;
    int cyclesSinceLastCheck = 0;
    for (int i = 0; i < cycles; i++)
    {
        if (i % (cycles/100) == 0)
        {
            Console.WriteLine(i + " Cycles completed, " + ((double)i/cycles) * 100 + "%  Avg Cost: " + cost / cyclesSinceLastCheck);
            cost = 0;
            cyclesSinceLastCheck = 0;
        }

        cyclesSinceLastCheck++;
        List<double> situation = new List<double>() { random.NextDouble() * 0.5, random.NextDouble() * 0.5 };
        
        double expected = situation[0] + situation[1];
        cost += NeuralNet.Cost(new List<double>() {expected}, testNeuralNet.Ask(situation));
        Gradient gradient = new Gradient(testNeuralNet, situation, new List<double>() {expected});
        gradient.ApplyGradient(testNeuralNet, 1);
    }
    Console.WriteLine("training complete");
    Console.WriteLine(testNeuralNet.ToString());
    while (true)
    {
        Console.WriteLine("first double number to add (> 0.5)");
        string input = Console.ReadLine();
        double inputDoubleOne;
        if (double.TryParse(input, out inputDoubleOne))
        {
            Console.WriteLine("second double number to add (> 0.5)");
            input = Console.ReadLine();
            double inputDoubleTwo;
            if (double.TryParse(input, out inputDoubleTwo))
            {
                List<double> output = testNeuralNet.Ask(new List<double>() { inputDoubleOne, inputDoubleTwo });
                Console.WriteLine("NeuralNet answer: " + output[0] + " Delta from correct answer: " + (inputDoubleOne + inputDoubleTwo - output[0]));
            }
        }
        
    }
    
    
    
    
}




class NeuralNet
{
    //A node is a set of connections, A layer is a set of nodes, a network is a set of layers
    public List<List<Node>> Network;

    public NeuralNet(int inputs, int outputs, List<int> hiddenLayerSizes, Random random)
    {
        Network = new List<List<Node>>();
        //Inputs
        Network.Add(CreateLayer(0, inputs,  random));
        
        //Hidden Layers
        for (int i = 0; i < hiddenLayerSizes.Count; i++)
        {
            Network.Add(CreateLayer(i==0 ? inputs : hiddenLayerSizes[i - 1], hiddenLayerSizes[i], random));
        }
        
        //Outputs
        Network.Add(CreateLayer(hiddenLayerSizes.Count > 0 ? hiddenLayerSizes[^1] : inputs, outputs,  random));
        
        //Connecting them all
        for (int i = 0; i < Network.Count - 1; i++)
        {
            UnboundLayerAttach(Network[i], Network[i+1]);
        }
    }

    static void UnboundLayerAttach(List<Node> Back, List<Node> Front)
    {
        for (int i = 0; i < Back.Count; i++)
        {
            for (int j = 0; j < Front.Count; j++)
            {
                Back[i].Front.Add(Front[j].Back[i]);
                Front[j].Back[i].SetToFrom(Front[j], Back[i]);
            }
        }
    }

    static List<Node> CreateLayer(int prevLayerSize, int layerSize, Random random)
    {
        //Only adding the backwards facing connections because we connect them, so to stack the layers only one 
        //side is needed per layer
        List<Node> layer = new List<Node>();
        for (int j = 0; j < layerSize; j++)
        {
            List<Connection> front = new List<Connection>();
            List<Connection> back = new List<Connection>();
            for (int k = 0; k < prevLayerSize; k++)
            {
                back.Add(new Connection(random));
            }
            if(prevLayerSize != 0)
                layer.Add(new Node(back, front, random));
            else
                layer.Add(new Node(back, front, 0));
        }

        return layer;
    }

    public List<double> Ask(List<double> situation)
    {
        
        if (situation.Count != Network[0].Count)
        {
            throw new Exception("Incorrect number of inputs");
        }

        Vector<double> currentLayerSituation = Vector<double>.Build.DenseOfArray(situation.ToArray());
        
        for (int i = 0; i < Network.Count - 1; i++)
        {
            Vector<double> biases = Vector<double>.Build.DenseOfArray(Network[i + 1].Select(node => node.Bias).ToArray());

            Matrix<double> weightMatrix =
                Matrix<double>.Build.DenseOfRows(Network[i].Select(node =>
                    node.Front.Select(connection => connection.Weight).ToList()).ToList());

            currentLayerSituation = (currentLayerSituation * weightMatrix + biases).Map(x => 1 / (1 + Math.Exp(-x)));
        }

        return currentLayerSituation.ToList();
    }
    
    public (List<List<double>> aValues, List<List<double>> zValues) AskFull(List<double> situation)
    {
        List<List<double>> finalZ = new List<List<double>>();
        finalZ.Add(situation);
        List<List<double>> finalA = new List<List<double>>();
        finalA.Add(situation);
        if (situation.Count != Network[0].Count)
        {
            throw new Exception("Incorrect number of inputs");
        }

        Vector<double> currentLayerSituation = Vector<double>.Build.DenseOfArray(situation.ToArray());
        Vector<double> currentLayerZ;
        
        for (int i = 0; i < Network.Count - 1; i++)
        {
            Vector<double> biases = Vector<double>.Build.DenseOfArray(Network[i + 1].Select(node => node.Bias).ToArray());

            Matrix<double> weightMatrix =
                Matrix<double>.Build.DenseOfRows(Network[i].Select(node =>
                    node.Front.Select(connection => connection.Weight).ToList()).ToList());
            currentLayerZ = (currentLayerSituation * weightMatrix + biases);
            currentLayerSituation = currentLayerZ.Map(x => 1 / (1 + Math.Exp(-x)));
            finalZ.Add(currentLayerZ.ToList());
            finalA.Add(currentLayerSituation.ToList());
        }

        return (finalA, finalZ);
    }

    public static double Cost(List<double> expected, List<double> received)
    {
        double final = 0;
        for (int i = 0; i < expected.Count; i++)
        {
            final += Math.Pow(received[i] - expected[i], 2);
        }

        return final;
    }

    public void Train(List<double> situation, List<double> expected, double stepMulti)
    {
        Gradient gradient = new Gradient(this, situation, expected);
        gradient.ApplyGradient(this, stepMulti);
        
    }
    public override string ToString()
    {
        string final = "";
        for (int l = 0; l < Network.Count(); l++)
        {
            final += "\nLayer: " + l;
            for (int k = 0; k < Network[l].Count; k++)
            {
                final += "\nNode: " + k + "\n Weights: ";
                for (int j = 0; j < Network[l][k].Front.Count(); j++)
                {
                    final += Network[l][k].Front[j].Weight + ", ";
                }

                final += "\n Bias: " + Network[l][k].Bias;
            }
        }
        
        return final;
    }
}

class Connection
{
    //this is storing more data than is absolutely needed, but idc tbh
    public double Weight;
    public Node To;
    public Node From;

    public Connection(double weight, Node to, Node from)
    {
        Weight = weight;

        To = to;
        From = from;
    }

    public Connection(Random random)
    {
        Weight = (random.NextDouble() - 0.5) * 10;
    }

    public void SetToFrom(Node to, Node from)
    {
        To = to;
        From = from;
    }
    
    

}

class Node
{
    public List<Connection> Back;
    public List<Connection> Front;
    public double Bias;

    public Node(List<Connection> back, List<Connection> front, Random random)
    {
        Back = back;
        Front = front;
        Bias = (random.NextDouble() - 0.5) * 10;
    }
    public Node(List<Connection> back, List<Connection> front, double bias)
    {
        Back = back;
        Front = front;
        Bias = bias;
    }

    public bool FinalLayer()
    {
        return Front.Any();
    }
    public bool FirstLayer()
    {
        return Back.Any();
    }
    
}

class Gradient
{
    public List<(List<List<double>> WeightDerivatives, List<double> BiasDerivatives)> GradientValueTuples;

    public Gradient(NeuralNet neuralNet, List<double> situation, List<double> expected)
    {
        GradientValueTuples = new List<(List<List<double>> WeightDerivatives, List<double> BiasDerivatives)>();
        
        (List<List<double>> aValues, List<List<double>> zValues) received =  neuralNet.AskFull(situation);
        List<double> dCostOverDValuesAbove = new List<double>();
        List<double> dCostOverDValues = new List<double>();
        // initializing the tuple array
        GradientValueTuples.Add((new List<List<double>>(),new List<double>()));
        for (int l = 1; l < neuralNet.Network.Count(); l++) 
        {
            GradientValueTuples.Add((new List<List<double>>(),new List<double>()));
            for (int k = 0; k < neuralNet.Network[l].Count; k++)
            {
                GradientValueTuples[l].WeightDerivatives.Add(new List<double>());
            }
        }
        //output layer
        for (int j = 0; j < received.aValues[^1].Count; j++) 
        {
            dCostOverDValuesAbove.Add(2 * (received.aValues[^1][j] - expected[j]));
        }
        
        for (int l = neuralNet.Network.Count - 1; l > 0; l--) // each non-input layer
        {
            
            for (int k = 0; k < neuralNet.Network[l][0].Back.Count; k++) // each node in the layer below
            {
                
                for (int j = 0; j < neuralNet.Network[l].Count; j++) //each node in the current layer
                {

                    //derivative of cost with respect to weight
                    double z = received.zValues[l][j];
                    double dAOverDZ = (1 / (1 + Math.Exp(-z))) * (1 - (1 / (1 + Math.Exp(-z))));
                    GradientValueTuples[l].WeightDerivatives[j].Add(received.aValues[l - 1][k] * dAOverDZ * dCostOverDValuesAbove[j]);

                    if (k == 0)
                    {
                        double biasDerivative = Math.Pow(1 + double.Exp(-z), -2) * double.Exp(-z) * dCostOverDValuesAbove[j];
                        //derivative of cost with respect to bias 
                        GradientValueTuples[l].BiasDerivatives.Add(biasDerivative);
                    }
                    
                    
                }

            }

            for (int k = 0; k < neuralNet.Network[l - 1].Count; k++) // each node in the layer 
            {
                double dCostOverDValue = 0;
                
                for (int j = 0; j < neuralNet.Network[l].Count; j++)// node in the layer above
                {
                    double z = received.zValues[l][j];
                    dCostOverDValue += neuralNet.Network[l][j].Back[k].Weight * (1 / (1 + Math.Exp(-z))) * (1 - (1 / (1 + Math.Exp(-z)))) * dCostOverDValuesAbove[j];
                }
                
                
                dCostOverDValues.Add(dCostOverDValue);
            }

            
            

            dCostOverDValuesAbove = dCostOverDValues;
            dCostOverDValues = new List<double>();
        }
        
        
        
        
    }

    public void ApplyGradient(NeuralNet neuralNet, double stepMulti)
    {
        for (int l = 1; l < neuralNet.Network.Count; l++) //layers
        {
            
                for (int k = 0; k < neuralNet.Network[l].Count; k++)//first node
                {

                    for (int j = 0; j < neuralNet.Network[l][k].Back.Count; j++) //second node
                    {
                        neuralNet.Network[l][k].Back[j].Weight -=
                            GradientValueTuples[l].WeightDerivatives[k][j] * stepMulti;
                        neuralNet.Network[l][k].Bias -= GradientValueTuples[l].BiasDerivatives[k] * stepMulti;
                    }
                    
                    
                    
                    
                }
            
                
            
        }
    }
    
    
    
}

abstract class Game
{
    public NeuralNet NeuralNet;

    public abstract void PrintGameState();
    public abstract void HumanTurnPlay();
    public abstract List<double> AIAction();
    public abstract void AITurnPlay();
    public abstract void PlayGame(int games, int printEvery);

}

class TicTacToe : Game
{
    private int[] gameState;
    private List<List<double>> gameHistory;
    public bool Player1Human;
    public bool Player2Human;
    private List<List<double>> Player1Moves;
    private List<List<double>> Player2Moves;

    public TicTacToe(bool player1Human, bool player2Human)
    {
        Player1Human = player1Human;
        Player2Human = player2Human;
        NeuralNet = new NeuralNet(19, 9, new List<int>() { 18, 18, 9, 9, 9, 9 }, new Random());
        gameState = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0 ,0 , 0 , 0, 0 , 0 , 0, 0 , 0 , 1};
        Player1Moves = new List<List<double>>();
        Player2Moves = new List<List<double>>();
        gameHistory = new List<List<double>>();
    }

    public override void PrintGameState()
    {
        Console.Clear();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if(j==1|| j==2)
                    Console.Write("|");
                if(gameState[(i*3) + j] == 0 && gameState[(i*3) + j + 9] == 0)
                    Console.Write((i*3) + j + 1);
                else if(gameState[(i*3) + j] == 1)
                    Console.Write("X");
                else if(gameState[(i*3) + j + 9] == 1)
                    Console.Write("O");
            }
            if(i != 2)
                Console.WriteLine("\n_____");
        }
        Console.WriteLine();
    }
    public override void HumanTurnPlay()
    {
        PrintGameState();
        Console.WriteLine("Your turn. Type the number of the square you want to play in.");
        string? input = Console.ReadLine();
        if (input != null && int.TryParse(input, out var playedPosition) && playedPosition < 10 && gameState[playedPosition - 1] == 0 && gameState[playedPosition + 8] == 0 )
        {
            gameHistory.Add(gameState.ToList().Select(pos => (double)pos).ToList());
            List<double> playedPositionList = new List<double>() {0,0,0,0,0,0,0,0,0};
            playedPositionList[playedPosition - 1] = 1;
            if (gameState[^1] == 1)
            {
                Player1Moves.Add(playedPositionList);
                gameState[playedPosition - 1] = 1;
                gameState[^1] = 0;
            }
            else
            {
                Player2Moves.Add(playedPositionList);
                gameState[^1] = 1;
                gameState[playedPosition + 8] = 1;
            }
        }
        else
        {
            Console.WriteLine("Input was not quite right, try again.");
            Thread.Sleep(100);
            HumanTurnPlay();
        }
        
    }

    public override void AITurnPlay()
    {
        List<double> aiAction = AIAction();
        
        
        if (gameState[^1] == 1)
        {
            gameHistory.Add(gameState.ToList().Select(pos => (double)pos).ToList());
            gameState[aiAction.IndexOf(aiAction.Max())] = 1;
            Player1Moves.Add(aiAction);
            gameState[^1] = 0;
        }
        else
        {
            gameHistory.Add(gameState.ToList().Select(pos => (double)pos).ToList());
            gameState[aiAction.IndexOf(aiAction.Max()) + 9] = 1;
            Player2Moves.Add(aiAction);
            gameState[^1] = 1;
        }

    }
    public override List<double> AIAction()
    {
        List<double> position = gameState.ToList().Select(p => (double)p).ToList();
        
        List<double> action = NeuralNet.Ask(position);
        for (int i = 0; i < 9; i++)
        {
            action[i] = position[i] == 0 && position[i + 9] == 0 ? action[i] : 0;
        }

        return action;
    }

    private int GameOver()
    {
        int gameOver = 0;
        if (gameState[0] == gameState[1] && gameState[0] == gameState[2] && gameState[0] != 0)
            gameOver = 1;
        else if (gameState[3] == gameState[4] && gameState[3] == gameState[5]&& gameState[3] != 0)
            gameOver = 1;
        else if (gameState[6] == gameState[7] && gameState[6] == gameState[8]&& gameState[6] != 0)
            gameOver = 1;
        else if (gameState[0] == gameState[3] && gameState[0] == gameState[6]&& gameState[0] != 0)
            gameOver = 1;
        else if (gameState[1] == gameState[4] && gameState[1] == gameState[7]&& gameState[1] != 0)
            gameOver = 1;
        else if (gameState[2] == gameState[5] && gameState[2] == gameState[8]&& gameState[2] != 0)
            gameOver = 1;
        else if (gameState[0] == gameState[4] && gameState[0] == gameState[8]&& gameState[0] != 0)
            gameOver = 1;
        else if (gameState[2] == gameState[4] && gameState[2] == gameState[6]&& gameState[2] != 0)
            gameOver = 1;
        
        else if (gameState[9] == gameState[10] && gameState[9] == gameState[11]&& gameState[9] != 0)
            gameOver = 2;
        else if (gameState[12] == gameState[13] && gameState[12] == gameState[14]&& gameState[12] != 0)
            gameOver = 2;
        else if (gameState[15] == gameState[16] && gameState[15] == gameState[17]&& gameState[15] != 0)
            gameOver = 2;
        else if (gameState[9] == gameState[12] && gameState[9] == gameState[15]&& gameState[9] != 0)
            gameOver = 2;
        else if (gameState[10] == gameState[13] && gameState[10] == gameState[16]&& gameState[10] != 0)
            gameOver = 2;
        else if (gameState[11] == gameState[14] && gameState[11] == gameState[17]&& gameState[11] != 0)
            gameOver = 2;
        else if (gameState[9] == gameState[13] && gameState[9] == gameState[17]&& gameState[9] != 0)
            gameOver = 2;
        else if (gameState[11] == gameState[13] && gameState[11] == gameState[15]&& gameState[11] != 0)
            gameOver = 2;
        

        if (gameOver != 0)
        {
            if (Player1Human || Player2Human)
            {
                Console.WriteLine("\n GAME OVER");
                Console.WriteLine($"Player {gameOver} won.");
                if (Player1Human && gameOver == 1 || Player2Human && gameOver == 2 && Player1Human != Player2Human)
                {
                    Console.WriteLine("Congrats, that's you!");
                }
            }

            for (int i = 0; i < Player1Moves.Count; i++)
            {
                if (gameOver == 1)
                {
                    NeuralNet.Train(gameHistory[i * 2], Player1Moves[i], 1);
                }
                else
                {
                    List<double> dontMoveThereAgain = gameHistory[i * 2];
                    dontMoveThereAgain[dontMoveThereAgain.IndexOf(dontMoveThereAgain.Max())] = 0;
                    NeuralNet.Train(dontMoveThereAgain, Player1Moves[i], -1);
                }
            }
            for (int i = 0; i < Player2Moves.Count; i++)
            {
                if (gameOver == 2)
                {
                    NeuralNet.Train(gameHistory[i * 2 + 1], Player2Moves[i], 1);
                }
                else
                {
                    List<double> dontMoveThereAgain = gameHistory[i * 2 + 1];
                    dontMoveThereAgain[dontMoveThereAgain.IndexOf(dontMoveThereAgain.Max())] = 0;
                    NeuralNet.Train(dontMoveThereAgain, Player1Moves[i], 1);
                }
            }

            return gameOver;
        }

        bool catsGame = true;

        for (int i = 0; i < 9; i++)
        {
            if (gameState[i] == 0 && gameState[i + 9] == 0)
            {
                catsGame = false;
            }
        }

        if (catsGame)
        {
            if (Player1Human || Player2Human)
            {
                Console.WriteLine("\n GAME OVER");
                Console.WriteLine($"Cat's Game");
                Thread.Sleep(300);
            }

            for (int i = 1; i < Player2Moves.Count + 1; i++)
            {
                List<double> dontMoveThereAgain = gameHistory[i * 2 - 1];
                dontMoveThereAgain[dontMoveThereAgain.IndexOf(dontMoveThereAgain.Max())] = 0;
                NeuralNet.Train(dontMoveThereAgain, Player1Moves[i], 0.001);
                dontMoveThereAgain = gameHistory[i * 2];
                dontMoveThereAgain[dontMoveThereAgain.IndexOf(dontMoveThereAgain.Max())] = 0;
                NeuralNet.Train(gameHistory[i*2], Player1Moves[i], 0.001);
                
            }


            return 3;
        }

        return 0;

    }

    public override void PlayGame(int games, int printEvery)
    {
        int p1wins = 0;
        int p2wins = 0;
        int cats = 0;
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        for (int i = 0; i < games; i++)
        {
            if (i % printEvery == 0 && i != 0)
            {
                stopwatch.Stop();
                Console.WriteLine($"Game {i} of {games} ({double.Round(((double)i/games) * 100,3)}%) " +
                                  $"in {stopwatch.Elapsed.Seconds.Round(2)} seconds ({(double)(games - i)/printEvery * stopwatch.Elapsed} remaining)\n " +
                                  $"p1 Winrate: {double.Round(((double)p1wins/printEvery) * 100,3)}% " +
                                  $"p2 Winrate: {double.Round(((double)p2wins/printEvery) * 100,3)}% " +
                                  $"Cats:  {double.Round(((double)cats/ printEvery) * 100,3)}% ");
                p1wins = 0;
                p2wins = 0;
                cats = 0;
                stopwatch.Restart();
            }
            gameState = new[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0,0,0,0,0,0,0,0,0, 1};
            gameHistory = new List<List<double>>();
            Player1Moves = new List<List<double>>();
            Player2Moves = new List<List<double>>();
            int gameOver = 0;
            
            while (GameOver() == 0)
            {
                if (gameState[^1] == 1)
                {
                    if (Player1Human)
                    {
                        HumanTurnPlay();
                    }
                    else
                    {
                        AITurnPlay();
                    }
                }
                else
                {
                    if (Player2Human)
                    {
                        HumanTurnPlay();
                    }
                    else
                    {
                        AITurnPlay();
                    }
                }

                gameOver = GameOver();
                if (gameOver == 1)
                {
                    p1wins++;
                }
                else if (gameOver == 2)
                {
                    p2wins++;
                }
                else if(gameOver == 3)
                {
                    cats++;
                }
            }
            
            
        }
    }
    
    
}
