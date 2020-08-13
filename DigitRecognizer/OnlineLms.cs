/* Ted Monchamp
 * CS830 - Artificial Intelligence
 * A10 - Digit Recognizer
 *
 * OnlineLms.cs - Online linear regression algorithm (uses one LmsClassifier for each label)
 */

using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace DigitRecognizer
{
    public static class OnlineLms
    {
        // IO handled inside the function
        public static void Run()
        {
            string settingsBuffer = Console.ReadLine();
            string[] settings = settingsBuffer.Split(' ');

            // 0, 2, and 4 have the parameters
            // 4 attributes, 4 values, 10 classes
            int attrCount = int.Parse(settings[0]);
            int valueCount = int.Parse(settings[2]);
            int classes = int.Parse(settings[4]);

            List<LmsClassifier> classifiers = new List<LmsClassifier>();

            for (int k = 0; k < classes; k++)
            {
                LmsClassifier lmcs = new LmsClassifier(k, attrCount);
                classifiers.Add(lmcs);
            }

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

                        // train
                        for (int c = 0; c < classes; c++)
                        {
                            classifiers[c].Learn(int.Parse(entry[0]), set);
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
                            double result = classifiers[c].Classify(set);
                            if (result > probability)
                            {
                                bestLabel = c;
                                probability = result;
                            }

                            totalResult += result;
                        }

                        double confidence = probability / totalResult;

                        Console.WriteLine("{0} {1:f3}", bestLabel, confidence);
                    }
                }
            }
        }
    }
}

