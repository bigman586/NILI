package com.example.bigman586.plug;

import android.content.Context;
import android.text.TextUtils;
import android.widget.Toast;

import java.math.BigDecimal;
import java.math.RoundingMode;

public final class Utilities {

    /**
     * Rounds double value to number specified
     * @return rounded double
     */
    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    /**
     * Creates toast with message specified
     */
    public static void errorToast(String message, Context context){
        Toast.makeText(context,
                message, Toast.LENGTH_SHORT).show();
    }

    /**
     * Check if entry from textfield is valid
     * @return boolean value
     */
    public static boolean isTextValid(String entry){
        if (!entry.matches("[a-zA-Z0-9.? ]*")){
            return false;
        }
        if (TextUtils.isDigitsOnly(entry)){
            return false;
        }
        if (entry.trim().matches("")){
            return false;
        }

        return true;
    }
}
