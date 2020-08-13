/* Ted Monchamp
 * CS830 - Artificial Intelligence
 * A10 - Digit Recognizer
 *
 * NaiveBayes.cs - classifier using naive bayesian approach
 *                 (no helper class - the model is basically just the big 3D array)
 */

using System;

namespace DigitRecognizer
{
    public static class NaiveBayes
    {
        public static void Run()
        {
            string settingsBuffer = Console.ReadLine();
            string[] settings = settingsBuffer.Split(' ');

            // 0, 2, and 4 have the parameters
            // 4 attributes, 4 values, 10 classes
            int attrCount = int.Parse(settings[0]);
            int valueCount = int.Parse(settings[2]);
            int classes = int.Parse(settings[4]);

            int[] classCounts = new int[classes];

            // right, 3D arrays are described differently in C#
            // use commas instead of brackets because jagged arrays are something VERY different
            int[,,] attributes = new int[classes, attrCount, valueCount];

            int trainingSetSize = 0;

            bool isTraining = false;
            bool isTesting = false;

            while (true)
            {
                string line = Console.ReadLine();

                if (string.IsNullOrEmpty(line))
                    break;

                if (line == "-- training --")
                {
                    isTraining = true;
                }
                else if (line == "-- test --")
                {
                    isTraining = false;
                    isTesting = true;
                }
                else
                {
                    if (isTraining)
                    {
                        string[] entry = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                        int[] set = new int[attrCount];
                        for (int k = 0; k < attrCount; k++)
                            set[k] = int.Parse(entry[k + 1]);

                        trainingSetSize++;

                        int label = int.Parse(entry[0]);
                        classCounts[label]++;

                        // train
                        for (int k = 0; k < attrCount; k++)
                        {
                            attributes[label, k, set[k]]++;
                        }
                    }
                    else if (isTesting)
                    {
                        string[] entry = line.Split(' ');

                        int[] set = new int[attrCount];
                        for (int k = 0; k < attrCount; k++)
                            set[k] = int.Parse(entry[k]);

                        int bestLabel = 0;
                        double probability = 0;
                        double totalResult = 0;

                        // classify item
                        for (int c = 0; c < classes; c++)
                        {
                            double result = GetProbability(c, set, attributes, classCounts, trainingSetSize);

                            if (result > probability)
                            {
                                bestLabel = c;
                                probability = result;
                            }

                            totalResult += result;
                        }

                        // P(D)P(D|H) / P(D)*sum_i(P_i(D|H))
                        // they all have the same P(D) because it's the same label, so don't bother calculating it
                        // all we need is the confidence in the label
                        double confidence = probability / totalResult;

                        Console.WriteLine("{0} {1:f3}", bestLabel, confidence);
                    }
                }
            }
        }

        // "probability" except not really, because I'm saving exec time by cutting out the P(D) term,
        // as it gets cancelled out of the confidence calculation (same for all outputs)
        private static double GetProbability(
            int label, int[] set, int[,,] attributes, int[] classCounts, int trainingSetSize)
        {
            // remember to use +1 smoothing (only for data - assume data can only fit a trained class)

            // P(H)
            double ph = (double)classCounts[label] / trainingSetSize;

            // accumulate all probabilities into product - naive approximation of P(D|H)
            double pdh = ph; // base case
            if (pdh > 0)
            {
                for (int k = 0; k < set.Length; k++)
                {
                    // I'm starting off with this cast as a double to head off int truncation issues
                    double count = attributes[label, k, set[k]] + 1d; // +1 smoothing
                    pdh *= count / classCounts[label];
                }
            }

            return pdh;
        }
    }
}

