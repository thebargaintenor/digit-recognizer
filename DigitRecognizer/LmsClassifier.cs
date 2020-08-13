/* Ted Monchamp
 * CS830 - Artificial Intelligence
 * A10 - Digit Recognizer
 *
 * LmsClassifier.cs - single classifier for linear regression algorithm
 */

namespace DigitRecognizer
{
    public class LmsClassifier
	{
		public int Label;
		public double[] Theta;

        const double LEARNING_RATE = 0.0001;

        private int _n = 0;

		public LmsClassifier (int label, int dimensions)
		{
			Label = label;
			Theta = new double[dimensions + 1];
		}

		public void Learn(int label, int[] set)
		{
			_n++;
			
			int y = (label == this.Label) ? 1 : 0;

			double yhat = Theta[0];
			for (int k = 0; k < set.Length; k++)
			{
				yhat += Theta[k + 1] * set[k];
			}

			// I don't actually remember why we changed the learning rate - this was almost 5 years ago
			// I *think* the starting rate was too high for small sets
			double alpha = LEARNING_RATE; //10d / (10d + _n);

			// go back and reread LMS regression for a better reason
			// I remember the value for the base case being 1, but uh... why.
			Theta[0] = Theta[0] - alpha * (yhat - y); // * 1, because augment each entry with 1

			for (int k = 0; k < set.Length; k++)
			{
				Theta[k + 1] = Theta[k + 1] - alpha * (yhat - y) * set[k];
			}
		}

		public double Classify(int[] set)
		{
			double yhat = Theta[0];
			for (int k = 0; k < set.Length; k++)
			{
				yhat += Theta[k + 1] * set[k];
			}

			return yhat;
		}
	}
}

