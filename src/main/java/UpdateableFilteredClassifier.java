//import adapted.HoeffdingTreeAdapted;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class UpdateableFilteredClassifier extends FilteredClassifier implements UpdateableClassifier {

	
	private static final long serialVersionUID = -5156208883093206801L;

	public UpdateableFilteredClassifier() {
		m_Classifier = new HoeffdingTree();
	    m_Filter = new weka.filters.supervised.attribute.Discretize();
	}
	
	@Override
	public void setClassifier(Classifier newClassifier){
		if(newClassifier instanceof UpdateableClassifier){
			m_Classifier = newClassifier;
		}
		else{
			System.out.println("Classifier has to be updateable");
		}
	}
	
	@Override
	public void updateClassifier(Instance instance) throws Exception {
		Instance filtered = super.filterInstance(instance);
		((UpdateableClassifier)super.getClassifier()).updateClassifier(filtered);
	}

	public void changeHoeffdingHeader(Instances instances, String labelname) throws Exception{
		//System.out.println("change Hoeffding header");
		String classname = instances.classAttribute().name();
		System.out.println("classname "+classname);
		Instances filtered = Filter.useFilter(instances, this.m_Filter);
		System.out.println("filtered num atts "+filtered.numAttributes());
		filtered.setClassIndex(filtered.attribute(classname).index());
//		((HoeffdingTreeAdapted)this.m_Classifier).setHeader(filtered, labelname);
	}

}
