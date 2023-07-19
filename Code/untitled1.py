# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:21:25 2023

@author: Ibrah
"""

def compile_patient_recom(static_features, cfi, threshold=0.1):

    
    patients = cfi.keys()
    fields = {'demographics':{'V1_OCCUPATION':['bin'], 
                                'V1_LIVING_ALONE':['bin'],'V1_DWELLING':['cat'],
                                'V1_INCOME_SRC':['cat'],'V1_LIVING_INVIRONMT':['cat']},
                     'somatic':{'V1_PE_WEIGHT_KG':['con'],'V1_PE_WAIST_CM':['con'],
                            'V1_PE_SBP':['con'],'V1_PE_DBP':['con'],'BMI':['con']}}
    names = {'V1_OCCUPATION':'work status',
             'V1_LIVING_ALONE':'living alone condition','V1_DWELLING':'Type of dwelling',
             'V1_INCOME_SRC':'main source of income','V1_LIVING_INVIRONMT':'living environment',
             'V1_PE_WEIGHT_KG':'weight','V1_PE_WAIST_CM':'waist','V1_PE_SBP':'systolic blood pressure',
             'V1_PE_DBP':'diastolic blood pressure','BMI':'BMI'}
    values = {'V1_OCCUPATION':["'NO'","'YES'"],
             'V1_LIVING_ALONE':["'NO'","'YES'"],'V1_DWELLING':["'HOUSE'","'APARTMENT'","'ROOM'",
                                                               "'HOUSE with supervision'","'DORMITORY'",
                                                               "'TEMPORARY'","'OTHER'"],
             'V1_INCOME_SRC':["'EMPLOYMENT'","'PARENTS'","'SOCIAL SECURITY'","'UNEMPLOYMENT BENEFIT'","'OTHER'"],
             'V1_LIVING_INVIRONMT':["'BIG CITY'","'MEDIUM CITY'","'SMALL CITY'","'VILLAGE'"],
             'V1_PE_WEIGHT_KG':[],'V1_PE_WAIST_CM':[],'V1_PE_SBP':[],'V1_PE_DBP':[],'BMI':[]}
    
    text = dict()
    vector = dict() 
    for patient in patients:
        text[patient] = ''
        vector[patient] = np.zeros([10,])
        sw = 0 
        for field in fields:
            for recom in fields[field]:
                if cfi[patient][field][recom]['effect_size']>=threshold:
                    if recom == 'V1_OCCUPATION':
                        if np.sum(cfi[patient][field][recom]['effect']) > 0:
                            text[patient] += """Your current work status is not optimal for your chances of remission. Consider whether it is possible to change it as it may improve your chances of remission by """ + \
                                "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '  
                            vector[patient][0] = 1
                        else:
                            text[patient] += """Maintaining your current work status can help to improve your chances of remission by """ + \
                            "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. ' 
                            vector[patient][0] = -1
                    elif recom == 'V1_LIVING_ALONE':
                        if (np.sum(cfi[patient][field][recom]['effect']) > 0 and 
                            static_features[field].loc[patient][recom]) == 2:
                            text[patient] += """Your current living situation may not be the best for achieving remission. Consider the possibility of living with someone as it may improve your chance for remission by """ + \
                                "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][1] = 1
                        elif np.sum(cfi[patient][field][recom]['effect']) <= 0:
                            text[patient] += """Your current living situation is putting you on a good track to remission. By maintaining it, the chance of remission increases by """ + \
                                "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '  
                            vector[patient][1] = -1
                    elif recom == 'V1_DWELLING':
                        if (cfi[patient][field][recom]['effect'][cfi[patient][field][recom]['index']]>0 and 
                            values[recom][int(static_features[field].loc[patient][recom]-1)]!=values[recom][cfi[patient][field][recom]['index']]):
                            text[patient] += 'Your current living environment in ' + values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ' may not be the best for achieving remission. Consider whether it is possible for you to live in a ' \
                                    + values[recom][cfi[patient][field][recom]['index']] + \
                                        ', as it may improve your chances for remission by  ' \
                                        + "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][2] = cfi[patient][field][recom]['index']
                        else:
                            text[patient] += 'By living in a ' + values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ', you are increasing your chances of remission by ' + \
                                 "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][2] = int(static_features[field].loc[patient][recom]-1)
                    elif recom == 'V1_INCOME_SRC':
                        if (cfi[patient][field][recom]['effect'][cfi[patient][field][recom]['index']]>0 and 
                            values[recom][int(static_features[field].loc[patient][recom]-1)]!=values[recom][cfi[patient][field][recom]['index']]):
                            text[patient] += 'Your main source of income at the moment (' + values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ') is not optimal. Consider the possibility to use ' \
                                    + values[recom][cfi[patient][field][recom]['index']] +', as it may improve your chances for remission by  ' \
                                        "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][3] = cfi[patient][field][recom]['index']
                        else:
                            text[patient] += 'Maintaining ' + values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ' as your main source of income can help you achieve remission by ' + \
                                 "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][3] = int(static_features[field].loc[patient][recom]-1)
                    elif recom == 'V1_LIVING_INVIRONMT':
                        if (cfi[patient][field][recom]['effect'][cfi[patient][field][recom]['index']]>0 and 
                            values[recom][int(static_features[field].loc[patient][recom]-1)]!=values[recom][cfi[patient][field][recom]['index']]):
                            text[patient] += """Sometimes moving to a different type of enviroment can help with recovery. Consider whether there is a possibility of temporarily (or permanently) moving to """ + \
                                    values[recom][cfi[patient][field][recom]['index']] + \
                                        ', as it may improve your chances for remission by ' + \
                                        "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][4] = cfi[patient][field][recom]['index']
                        else:
                            text[patient] += """Some living environments improve the chances of getting better sooner. By continuing to live in a """ + \
                                values[recom][int(static_features[field].loc[patient][recom]-1)] + \
                                ', your chances of remission are increased by ' + \
                                 "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '
                            vector[patient][4] = int(static_features[field].loc[patient][recom]-1)
                   
                    elif (recom == 'V1_PE_WEIGHT_KG'):
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            vector[patient][5] = 1
                            if sw == 0:
                                text[patient] += """Consider your eating habits at the moment and your lifestyle choices. Eating more wholesome, home-cooked meals, as well as moving more could improve your health and subsequently lead to increasing your chance of remission by """ + \
                                    "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '   
                                sw = 1
                            
                    elif (recom == 'V1_PE_WAIST_CM'):
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            vector[patient][6] = 1
                            if sw == 0:
                                text[patient] += """Consider your eating habits at the moment and your lifestyle choices. Eating more wholesome, home-cooked meals, as well as moving more could improve your health and subsequently lead to increasing your chance of remission by """ + \
                                    "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '   
                                sw = 1
                    elif (recom == 'V1_PE_SBP'):
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            vector[patient][7] = 1
                            if sw == 0:
                                text[patient] += """Consider your eating habits at the moment and your lifestyle choices. Eating more wholesome, home-cooked meals, as well as moving more could improve your health and subsequently lead to increasing your chance of remission by """ + \
                                    "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '   
                                sw = 1
                    elif (recom == 'V1_PE_DBP'):
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            vector[patient][8] = 1
                            if sw == 0:
                                text[patient] += """Consider your eating habits at the moment and your lifestyle choices. Eating more wholesome, home-cooked meals, as well as moving more could improve your health and subsequently lead to increasing your chance of remission by """ + \
                                    "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '   
                                sw = 1
                    elif (recom == 'BMI'):
                        if cfi[patient][field][recom]['effect'][0] > 0 or cfi[patient][field][recom]['effect'][-1] < 0:
                            vector[patient][9] = 1
                            if sw == 0:
                                text[patient] += """Consider your eating habits at the moment and your lifestyle choices. Eating more wholesome, home-cooked meals, as well as moving more could improve your health and subsequently lead to increasing your chance of remission by """ + \
                                    "{:.2f}".format(cfi[patient][field][recom]['effect_size']) + '. '   
                                sw = 1
                                 
        if text[patient]=='':
            text[patient] += 'The model has no recommendation for this patient.'
                
    return text, vector                      