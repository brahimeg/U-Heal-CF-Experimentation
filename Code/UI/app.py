import warnings, os, json, sys
if (os.getcwd()[-2:] == 'UI'):
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
if (os.getcwd()[-6:] == 'U-Heal' or os.getcwd()[-1:] == '/'):
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), 'Code')))
sys.path.append(os.getcwd())

from flask import Flask, render_template, request
from flask import *
from Carla.models.negative_instances import predict_negative_instances
from Utilities.customclf import CustomClf
from Carla.plotting.plotting import *
from Utilities.carla_utilities import *
from Datasets.optimize import compute_classifier_inputs
from Datasets.optimize import read_dynamic_features, feature_extraction
from Datasets.optimize import compute_output_labels, read_static_features 
from Carla.data.catalog import CsvCatalog
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

app = Flask(__name__)

if ('Code' in os.getcwd()):
    base_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data', 'Optimise')
else:
    base_path = os.path.join(os.getcwd(), 'Data', 'Optimise')
data_path = base_path + '/'
save_path = os.path.join(base_path, 'Results', 'Classifiers')
figure_path = os.path.join(base_path, 'Results', 'Figures')
carla_save_path = os.path.join(base_path, 'Results', 'Counterfactuals')

if not os.path.isdir(os.path.join(base_path, 'Results')):
    os.mkdir(os.path.join(base_path, 'Results'))
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
    
@app.route('/execute_best_cf', methods=['POST'])
def execute_best_cf():
    selected_rc_methods = request.json.get('selectedRCMethods')
    selected_metrics = request.json.get('selectedMetrics')
    selected_instance = request.json.get('selectedNegativeInstance')    
    cutt_off = request.json.get('nParameter')

    # read in default hyperparameters for recourse methods and adjust to account for single mode generation
    if ('Code' in os.getcwd()):
        hyper_params_path = os.path.join(os.getcwd(), 'Utilities', 'carla_hyper_parameters.json')
    else:
        hyper_params_path = os.path.join(os.getcwd(), 'Code', 'Utilities', 'carla_hyper_parameters.json')
    hyper_parameters = json.load(open(hyper_params_path))
    hyper_parameters['gower_cf']['retries'] = len(dataset.df) - len(factuals)
    hyper_parameters['gower_cf']['single_mode'] = True

    subject = int(selected_instance)
    all_results = single_generate_counterfactuals(model, hyper_parameters, factuals.loc[subject], int(cutt_off), selected_rc_methods)   

    # Rank generated counterfactuals using average rank method across different metrics
    cf, bench, ranks_df = return_best_cf(all_results, 1, selected_metrics)
    bench.columns = ['Method', 'Single-y-Nearest-Neighbours', 'Stability', 'Redundancy',
                    'Sparsity', 'L0-distance', 'L1-distance', 'L2-distance',
                    'Linf-distance', 'Avg-Time', 'Connectedness', 'Avg-Rank']
    if cf.empty:
        return jsonify({'message': "No Counterfactuals were generated!"})
    cf_unscaled, factuals_unscaled = transform_features_to_original_scale(cf, factuals, [subject], scalers)
    bench.index = [subject]
    cf.index = [subject]
    result_message = generate_recommendation(factuals_unscaled.loc[subject], cf_unscaled.loc[subject]) 
    result_message = f'Generated {len(ranks_df)} counterfactuals for subject {subject}\n\n' + 'Benchmarks highest ranked counterfactual:\n'+ bench.iloc[0].to_string() + '\n\nRecommendations:\n' + result_message
    return jsonify({'message': result_message})

@app.route('/qqplot_image')
def qqplot_image():
    qqplt_image_path = os.path.join(figure_path, 'qqplot.png')
    return send_file(qqplt_image_path, mimetype='image/png')

@app.route('/select_options', methods=['POST'])
def select_options():
    data = request.get_json()
    selected_rc_methods = data.get('selectedRCMethods', [])
    selected_metrics = data.get('selectedMetrics', [])

    # Perform actions based on selected RC methods and metrics
    response_message = f"Selected RC methods: {selected_rc_methods}, Selected metrics: {selected_metrics}"

    return jsonify({'message': response_message})

@app.route('/')
def index():
    return render_template('index.html', init_status="")

@app.route('/get_negative_instances_data', methods=['GET'])
def get_negative_instances_data():
    global factuals
    factuals = predict_negative_instances(model, dataset.df[model.X.columns])
    fetched_negative_instances = list(factuals.index)
    return jsonify({'negative_instances': fetched_negative_instances})

@app.route('/select_negative_instance', methods=['POST'])
def select_negative_instance():
    selected_instance = request.form.get('selected_instance')
    response_message = f"Selected negative instance: {selected_instance}"
    
    return jsonify({'message': response_message})

@app.route('/init_classifier', methods=['POST'])
def init_classifier():
    global dataset
    global model
    global scalers 
    
    selected_classifier = request.json.get('classifier')
    clfs = {"Logistic Regression": LogisticRegression(),
            "SVC Linear": SVC(kernel="linear", probability=True),
            "SVC RBF" : SVC(probability=True),
            "KNeighbors Classifier" :KNeighborsClassifier(),
            "MLP" : MLPClassifier(),
            "DecisionTree" : DecisionTreeClassifier(),
            "RandomForest" : RandomForestClassifier(),
            "Ada Boost" : AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "GaussianNB" : GaussianNB() }
    # Initialize the classifier based on the user's selection
    initialized_classifier = clfs[selected_classifier]

    # Add other classifier initialization logic here

        
    ################################## READING DATA ###############################
        
    # The visits used for the experiment (do not change this)
    visits = ['2','3','5']

    # The visits used as input to the classifier
    feature_visits = ['2']

    # The visit used as the output of classifier
    label_visit = '5'


    # Here specify the assessments for dynamic features
    assessments = ['PANNS', 'PSP', 'CGI']

    criteria = ['PANNS', 'PSP', 'CGI']

    # Here specify the demographics information you want to retrieve
    with open(os.path.join(data_path, 'daniel_static_features.json'),'r') as file_object:  
        static_feature_types = json.load(file_object)



    static_features = read_static_features(data_path, static_feature_types)
            
    dynamic_features = read_dynamic_features(data_path, visits, assessments)

    features, static_feature_types, scalers = feature_extraction(dynamic_features, 
                                                        static_features=static_features, 
                                                        static_feature_types=static_feature_types, 
                                                        aggregation='union')
    remissions = compute_output_labels(dynamic_features, labels=criteria)

    X, Y, subjects, X_df = compute_classifier_inputs(features, remissions, assessments,
                                                    'PANNS', static_feature_types, feature_visits, 
                                                    label_visit, 'impute')


    ############################### CLASSIFICATION ################################

    # Prepare and create carala dataset  
    X_df, continuous, categorical, immutable, translation_dict = determine_feature_types(X_df, static_feature_types, assessments, 
                                                                                        use_orginal_column_names=True)
    muttable = list(set(continuous+categorical) - set(immutable))
    X_df['Y'] = Y
    dataset = CsvCatalog(df=X_df,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='Y')

    # Create custom classifier using carla interface    
    model = CustomClf(dataset, clf=initialized_classifier, scaling=False, fit_full_data=True, calibration=None, bagging=50)
    response_message = f"Successfully initialized {selected_classifier}"
    return jsonify({'message': response_message})

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
