/* Ted Monchamp
 * CS830 - Artificial Intelligence
 * A10 - Digit Recognizer
 *
 * KnnClassifier.cs - classifier used by kNN algorithm (does most of the work)
 */

using System;
using System.Linq;
using System.Collections.Generic;

namespace DigitRecognizer
{
	public class KnnClassifier
	{
		private readonly int _k = 3; // hard code for now
		private readonly int _dimensions = 1;

		// Remember, the category is the first item in the set!
		// (I don't feel like introducing class overhead when everything
		// already has to stay in memory.)
		private readonly List<int[]> _trainingSet;

		public KnnClassifier(int dimensions)
        {
			_dimensions = dimensions;
			_trainingSet = new List<int[]>();
        }

		public void Add(int[] set)
		{
			_trainingSet.Add(set);
		}

		public (int, double) Classify(int[] set)
		{
			var kset = new List<Match>();

			for (int i = 0; i < _trainingSet.Count; i++)
			{
				var m = Distance(set, _trainingSet[i]);

				if (kset.Count > 0)
				{
					bool added = false;
					for (int j = 0; j < kset.Count; j++)
					{
						if (m.Distance < kset[j].Distance)
						{
							kset.Insert(j, m);
							added = true;
							break;
						}
					}

					if (!added && kset.Count < _k)
						kset.Add(m);

					// prune extra items
					if (kset.Count > _k)
						kset.RemoveRange(_k, kset.Count - _k);
				}
				else
					kset.Add(m);
			}

			var best = kset
				.GroupBy(k => k.Label)
				.OrderByDescending(g => g.Count())
				.First();

			double confidence = best.Count() / (double)kset.Count;

			return (best.Key, confidence);
		}

		private Match Distance (int[] test, int[] training)
		{
			double dsq = 0;
			for (int i = 0; i < _dimensions; i++)
			{
				dsq += (test[i] - training[i + 1]) * (test[i] - training[i + 1]);
			}

			return new Match
			{
				Label = training[0],
				Distance = Math.Sqrt(dsq)
			};
		}
	}

	struct Match
	{
		public int Label;
		public double Distance;
	}
}

