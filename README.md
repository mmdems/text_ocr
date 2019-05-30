# text_ocr
XMLs are containing objects (tag WordFragment). Objects have features (tags ВlackCount, WhiteHolesCount, HorzStrokesCount, VertStrokesCount and MaxHorzStrokeLength).

First the feature space is formed with the help XMLs. Then data is being splitted into test set and train set.

Non_text objects are labeled as 0, text objects - as 1.

Chosen model is random forest classifier.

At the end there are some graphs and computed metrics.
