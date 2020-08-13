/* Ted Monchamp
 * CS830 - Artificial Intelligence
 * A10 - Digit Recognizer
 *
 * Knn.cs - mostly the parser for k-NN algorithm (uses one KnnClassifier for whole algorithm)
 */

using System;

namespace DigitRecognizer
{
    public static class Knn
    {
        static public void Run()
        {
            string settingsBuffer = Console.ReadLine();
            string[] settings = settingsBuffer.Split(' ');

            // 0, 2, and 4 have the parameters
            // 4 attributes, 4 values, 10 classes
            int attrCount = int.Parse(settings[0]);

            KnnClassifier model = new KnnClassifier(attrCount);

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
                    continue;
                }

                if (line == "-- test --")
                {
                    isTraining = false;
                    isTesting = true;
                    continue;
                }

                if (isTraining)
                {
                    // training vectors need to keep their labels in position 0
                    var vector = ParseNInts(attrCount + 1, line);
                    model.Add(vector);
                }
                else if (isTesting)
                {
                    var vector = ParseNInts(attrCount, line);
                    var (label, confidence) = model.Classify(vector);

                    Console.WriteLine("{0} {1:f3}", label, confidence);
                }
            }
        }

        static int[] ParseNInts(int n, string input)
        {
            var entry = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            //Console.WriteLine("Found {0} items in row", entry.Length);

            var set = new int[n];
            for (int k = 0; k < n; k++)
                set[k] = int.Parse(entry[k]);

            return set;
        }
    }
}

