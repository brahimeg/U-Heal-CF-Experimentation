import warnings, os, json, threading
if (os.getcwd()[-2:] == 'UI'):
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import tkinter as tk
from tkinter import ttk, scrolledtext
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")

################################### PATHS ###################################    
if ('Code' in os.getcwd()):
    base_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data', 'Optimise')
else:
    base_path = os.path.join(os.getcwd(), 'Data', 'Optimise')
data_path = base_path + '/'
save_path = os.path.join(base_path, 'Results', 'Classifiers')
carla_save_path = os.path.join(base_path, 'Results', 'Counterfactuals')

if not os.path.isdir(os.path.join(base_path, 'Results')):
    os.mkdir(os.path.join(base_path, 'Results'))
if not os.path.isdir(save_path):
    os.mkdir(save_path)
     

def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))
    
def init_classifier():
    global dataset
    global model
    global scalers 
        
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
    model = CustomClf(dataset, clf=eval(classifier_combobox.get()), scaling=False, fit_full_data=True, calibration=None, bagging=50)
    get_neg_instances_button.config(state=tk.NORMAL)

def get_negative_instances():
    global factuals
    get_neg_instances_button.config(state=tk.DISABLED)
    best_cf_button.config(state=tk.NORMAL)
    factuals = predict_negative_instances(model, dataset.df[model.X.columns])
    for i in factuals.index:
        negative_instances_listbox.insert(i, str(i))
    

def execute_best_cf():
    best_cf_button.config(state=tk.DISABLED)
    chosen_rc_methods = [rc for rc, var in rc_method_vars.items() if var.get()]
    chosen_metrics = [metric for metric, var in metric_vars.items() if var.get()]
    
    # read in default hyperparameters for recourse methods and adjust to account for single mode generation
    if ('Code' in os.getcwd()):
        hyper_params_path = os.path.join(os.getcwd(), 'Utilities', 'carla_hyper_parameters.json')
    else:
        hyper_params_path = os.path.join(os.getcwd(), 'Code', 'Utilities', 'carla_hyper_parameters.json')
    hyper_parameters = json.load(open(hyper_params_path))
    hyper_parameters['gower_cf']['retries'] = len(dataset.df) - len(factuals)
    hyper_parameters['gower_cf']['single_mode'] = True

    subject = int(negative_instances_listbox.get(negative_instances_listbox.curselection()))
    output_text.insert(tk.INSERT, f"Subject: {subject}\n")
    output_text.insert(tk.INSERT, "Running.... \n")
    all_results = single_generate_counterfactuals(model, hyper_parameters, factuals.loc[subject], int(n_value.get()), chosen_rc_methods)   

    # Rank generated counterfactuals using average rank method across different metrics
    cf, bench, ranks_df = return_best_cf(all_results, 1, chosen_metrics)
    if cf.empty:
        output_text.insert(tk.END, "No CFs found for these settings! \n\n")
        best_cf_button.config(state=tk.NORMAL)
        return
    cf_unscaled, factuals_unscaled = transform_features_to_original_scale(cf, factuals, [subject], scalers)
    bench.index = [subject]
    cf.index = [subject]
    qqplt = single_sample_normality_test(cf, model)
    display_figure(qqplt, 19, 0)
    cf_probas = generate_confidence_intervals(cf, model.raw_model)
    original_probas = generate_confidence_intervals(factuals.loc[[subject]], model.raw_model)
    merged_probas = original_probas.join(cf_probas, lsuffix='_original', rsuffix='_cf')
    diff_vals, ssplt = single_sample_plot(factuals_unscaled.loc[subject], cf_unscaled.loc[subject], dataset, figsize=(5,2))
    display_figure(ssplt, 20, 0)
    # Print results
    output_text.insert(tk.END, diff_vals)
    output_text.insert(tk.END, "\n")
    output_text.insert(tk.END, bench.loc[subject])
    output_text.insert(tk.END, "\n")
    output_text.insert(tk.END, merged_probas.loc[subject])
    output_text.insert(tk.END, "\n\n")
    best_cf_button.config(state=tk.NORMAL)
    
def display_figure(fig, row=18, column=4):
    """
    Display a matplotlib figure in the tkinter window
    """
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame, )
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.config(width=150, height=250) 
    canvas_widget.grid(row=row, column=column,columnspan=4,pady=1, ipady=1, sticky=tk.N+tk.S+tk.E+tk.W)
    canvas.draw()

# ... (rest of your code)

def start_execute_best_cf():
    t = threading.Thread(target=execute_best_cf)
    t.start()
    
def start_init_classifier():
    t = threading.Thread(target=init_classifier)
    t.start()

root = tk.Tk()
root.title("CARLA Experiment UI")
root.geometry('700x800')

# Create a canvas inside the main window
canvas = tk.Canvas(root)
canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Add a vertical scrollbar to the main window, linked to the canvas
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
canvas.configure(yscrollcommand=scrollbar.set, width=680, height=780)

# Create a frame inside the canvas
frame = ttk.Frame(canvas, padding="15")
frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

# Bind the frame's size change to an update function
frame.bind("<Configure>", lambda e: onFrameConfigure(canvas))

# frame = ttk.Frame(root, padding="15")
# frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Step 1: Classifier Dropdown and Button
ttk.Label(frame, text="1. Select Classifier").grid(rowspan=1, column=0, row=0, padx=1, pady=1)
classifier_options = [LogisticRegression(),
                        SVC(kernel="linear", probability=True),
                        SVC(probability=True),
                        KNeighborsClassifier(),
                        MLPClassifier(),
                        DecisionTreeClassifier(),
                        RandomForestClassifier(),
                        AdaBoostClassifier(),
                        GradientBoostingClassifier(),
                        GaussianNB(),
                    ]
classifier_combobox = ttk.Combobox(frame, values=classifier_options, state="readonly")
classifier_combobox.grid(rowspan=1, column=0, row=1, padx=1, pady=1)
init_clf_button = ttk.Button(frame, text="Initialize Classifier", command=start_init_classifier)
init_clf_button.grid(rowspan=1, column=0, row=2, pady=1, padx=1)

# Step 2: Negative Instances Button and Listbox
get_neg_instances_button = ttk.Button(frame, text="2. Get Negative Instances", command=get_negative_instances, state=tk.DISABLED)
get_neg_instances_button.grid(rowspan=1, row=0, column=1,  pady=1, padx=1)
negative_instances_listbox = tk.Listbox(frame, exportselection=False)
negative_instances_listbox.grid(rowspan=4, row=1, column=1, pady=1, padx=1)

ttk.Label(frame).grid(column=0, row=6, padx=1, pady=1, sticky=tk.W)

# Step 3: RC Methods and Metrics
ttk.Label(frame, text="3. CF Methods:").grid(column=0, row=7, padx=1, pady=1, sticky=tk.W)
rc_methods = ["gs","gower_cf", "dice", "revise", "cchvae"]  
rc_method_vars = {method: tk.BooleanVar() for method in rc_methods}
for i, (method, var) in enumerate(rc_method_vars.items()):
    ttk.Checkbutton(frame, text=method, variable=var).grid(column=0, row=i+8, padx=1, pady=1, sticky=tk.W)
    
ttk.Label(frame, text="4. Metrics used for Ranking:").grid(column=1, row=7, padx=1, pady=1, sticky=tk.W)
metrics = ['L2_distance', 
            'L1_distance',
            'L0_distance',
            'Redundancy',
            'Sparsity',
            'avg_time',
            'Stability', 
            'single-y-Nearest-Neighbours'] 
metric_vars = {metric: tk.BooleanVar() for metric in metrics}
for i, (metric, var) in enumerate(metric_vars.items()):
    if i <= len(metric_vars.items())/2:
        ttk.Checkbutton(frame, text=metric, variable=var).grid(column=1, row=i+8, padx=1, pady=1, sticky=tk.W)
    else:
        temp = int(len(metric_vars.items())/2)+1
        ttk.Checkbutton(frame, text=metric, variable=var).grid(column=2, row=i+8-temp, padx=1, pady=1, sticky=tk.W)
        
ttk.Label(frame, text="Number of CFs per method:").grid(row=17, column=0, padx=1, pady=1, sticky=tk.W)
n_value = tk.StringVar()
n_entry = ttk.Entry(frame, textvariable=n_value)
n_entry.grid(row=17, column=1, padx=1, pady=1, sticky=tk.W)
n_entry.insert(0, "3")  

best_cf_button = ttk.Button(frame, text="5. Find best CF", command=start_execute_best_cf, state=tk.DISABLED)
best_cf_button.grid(columnspan=1,column=2, row=17, pady=20)

output_text = scrolledtext.ScrolledText(frame, width=70, height=20)
output_text.grid(columnspan=4, row=18, padx=5, pady=5)
        
root.mainloop()
