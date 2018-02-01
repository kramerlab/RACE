import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.regressor.transformation.TransformationBasedMultiTargetRegressor;
import mulan.transformations.regression.SingleTargetTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;


public class myMultiTargetbinaryRelevance extends TransformationBasedMultiTargetRegressor {

    /**
     * The ensemble of binary relevance models. These are Weka Classifier
     * objects.
     */
    protected Classifier[] ensemble;
    /**
     * The correspondence between ensemble models and labels
     */
    private String[] correspondence;
    protected SingleTargetTransformation brt;

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public myMultiTargetbinaryRelevance(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ensemble = new Classifier[numLabels];

        correspondence = new String[numLabels];
        for (int i = 0; i < numLabels; i++) {
            correspondence[i] = train.getDataSet().attribute(labelIndices[i]).name();
        }

        debug("preparing shell");
        brt = new SingleTargetTransformation(train);

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = AbstractClassifier.makeCopy(baseRegressor);
            Instances shell = brt.transformInstances(i);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(shell);
        }
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];


        for (int counter = 0; counter < numLabels; counter++) {
            Instance transformedInstance = brt.transformInstance(instance, counter);
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(transformedInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            bipartition[counter] = (maxIndex == 1) ? true : false;

            // The confidence of the label being equal to 1
            confidences[counter] = distribution[1];
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    /**
     * Returns the model which corresponds to the label with labelName
     *
     * @param labelName
     * @return the corresponding model or null if the labelIndex is wrong
     */
    public Classifier getModel(String labelName) {
        for (int i = 0; i < numLabels; i++) {
            if (correspondence[i].equals(labelName)) {
                return ensemble[i];
            }
        }
        return null;
    }

	@Override
	protected String getModelForTarget(int targetIndex) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}
}