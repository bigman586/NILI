package com.example.bigman586.plug;

public enum CustomPagerEnum {
    LABEL("Label", R.layout.activity_label),
    PREDICT("Predict", R.layout.activity_predict);

    private String mTitleResId;
    private int mLayoutResId;

    CustomPagerEnum(String titleResId, int layoutResId) {
        mTitleResId = titleResId;
        mLayoutResId = layoutResId;
    }

    public String getTitleResId() {
        return mTitleResId;
    }

    public int getLayoutResId() {
        return mLayoutResId;
    }

}
