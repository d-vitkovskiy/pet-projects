import datetime

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)


def create_labels(df):
    """
    Создаём из датасета объект Series с нужными метками
    """
    df = df.copy()
    target_events = [
        'sub_car_claim_click', 'sub_car_claim_submit_click',
        'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]
    df['target'] = df.event_action.isin(target_events).astype(int)
    grouped = (df.groupby('session_id')['target'].sum().map(lambda x: 1
                                                            if x > 0 else 0))
    return grouped


def filter_data(df):
    """
    Удаляем ненужные признаки
    """
    columns_to_drop = [
        'device_model',
        'session_id',
        'client_id',
        'device_screen_resolution',
    ]
    return df.drop(columns_to_drop, axis=1)


def obj_to_date(df):
    """
    Преобразуем дату и время из строкового формата
    в формат datetime
    """
    import pandas
    df = df.copy()
    df['visit_date'] = pandas.to_datetime(
        df['visit_date'].map(lambda x: str(x)))
    df['visit_time'] = pandas.to_datetime(
        df['visit_time'].map(lambda x: str(x)))
    return df


def new_features(df):
    """
    Создаём новые признаки
    """
    import pandas
    df = df.copy()
    df['visit_weekday'] = df.visit_date.dt.weekday.astype(int)
    df['visit_month'] = df.visit_date.dt.month.astype(int)
    df['visit_day'] = df.visit_date.dt.day.astype(int)
    df['visit_hour'] = df.visit_time.dt.hour.astype(int)
    df['new_visitor'] = df.visit_number.map(lambda x: 1 if x == 1 else 0)
    df['first_month'] = df.groupby('client_id').visit_month.transform('min')
    df['month_duration'] = df.apply(lambda x: x.visit_month - x.first_month,
                                    axis=1)

    organic = ['organic', 'referral', '(none)']

    df['organic'] = df.utm_medium.map(lambda x: ('Y' if x in organic else 'N')
                                      if not pandas.isna(x) else x)

    social = [
        'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
        'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
        'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
    ]

    df['social'] = df.utm_source.map(lambda x: ('Y' if x in social else 'N')
                                     if not pandas.isna(x) else x)
    return df


def main():

    print('Event action prediction for SberAuto subscription')

    # Загрузим данные из исходных датасетов
    sessions = pd.read_csv('data/ga_sessions.csv', sep=',')
    hits = pd.read_csv('data/ga_hits.csv', sep=',')
    # Создаём метки
    hits_group = create_labels(hits)
    # Формируем датасет с метками для дальнейшей работы
    df = pd.merge(sessions, hits_group, on='session_id', how='inner')

    X = df.drop('target', axis=1)
    y = df.target.copy()

    with open('xgb_model.pkl', 'rb') as file:
        clf = dill.load(file)

    add_features = Pipeline([
        ('date', FunctionTransformer(obj_to_date)),
        ('features', FunctionTransformer(new_features)),
        ('filter', FunctionTransformer(filter_data))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse=True, handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
        ('num', num_pipeline, make_column_selector(dtype_include=int))
    ])

    pipeline = Pipeline([
        ('add_features', add_features),
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X, y, classifier__verbose=False)

    with open('sber_auto_pipe.pkl', 'wb') as fh:
        dill.dump(
            {
                'model': pipeline,
                'metadata': {
                    'name':
                    'Event action prediction for SberAuto subscription',
                    'author': 'Dmitry Vitkovskiy',
                    'version': 4,
                    'date': datetime.datetime.now(),
                    'type': 'XGBClassifier'
                }
            }, fh)


if __name__ == '__main__':
    main()
