package com.example.bigman586.plug;

import android.app.ProgressDialog;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.RequiresApi;
import android.support.v4.app.Fragment;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * A screen that allows user to label data
 */
public class LabelFragment extends Fragment  {

    // UI references.
    /*private EditText mPasswordView;
    private View mProgressView;
    private View mLoginFormView;*/
    // Store instance variables
    private String title;
    private int page;

    private ImageButton submit;
    private AutoCompleteTextView labelField;
    private TextView currentDevice;
    private ProgressDialog progressBar;
    private String entry;
    private EditText placeHolder;

    private ArrayList<String> autoList;

    public static LabelFragment newInstance(int page, String title) {

        LabelFragment fragmentFirst = new LabelFragment();
        Bundle args = new Bundle();

        args.putInt("0", page);
        args.putString("Label", title);

        fragmentFirst.setArguments(args);
        return fragmentFirst;
    }

    // Store instance variables based on arguments passed
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        page = getArguments().getInt("someInt", 0);
        title = getArguments().getString("someTitle");
    }


    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        final ViewGroup rootView = (ViewGroup) inflater.inflate(
                R.layout.activity_label, container, false);

        //creates progress bar
        progressBar = new ProgressDialog(getActivity());
        progressBar.setMessage("Loading....");

        //Connect UI Elements to variables
        submit = rootView.findViewById(R.id.send);
        labelField = rootView.findViewById(R.id.labelField);

        autoList = new ArrayList<String>();
        ArrayAdapter<String> adapter;
        adapter = new ArrayAdapter<>(Objects.requireNonNull(getActivity()),
                android.R.layout.simple_dropdown_item_1line, autoList);

        labelField.setAdapter(adapter);
        placeHolder = rootView.findViewById(R.id.placeHolder);

        currentDevice = rootView.findViewById(R.id.currentDevice);
        currentDevice.setText("");

        fillLabelArray();

        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                submitLabel();
                Utilities.hideKeyboard(getContext(), getActivity());
                labelField.setText("");
                System.out.println("Button clicked");
            }
        });

            return rootView;
    }

    /**
     * Send label to server using retrofit
     */
    private void submitLabel() {
        this.entry = String.valueOf(labelField.getText());

        if (TextUtils.isEmpty(entry)) {
            placeHolder.setError(getString(R.string.error_field_required));
            placeHolder.requestFocus();
            return;
        }

        if (!Utilities.isTextValid(entry)) {
            Toast.makeText(getContext(),
                    (getString(R.string.error_wrong_label)), Toast.LENGTH_SHORT).show();
            return;
        }

        // shows progress bar
        progressBar.show();

        JsonObject jsonObj = new JsonObject();
        jsonObj.addProperty("label", entry);

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call call = service.postLabel(jsonObj);
        call.enqueue(new Callback() {

            @Override
            public void onFailure(Call call, Throwable t) {

                Utilities.errorToast(getString(R.string.error_no_connection), getContext());
                Utilities.errorToast(getString(R.string.error_no_send), getContext());
            }

            @Override
            public void onResponse(Call call, Response response) {

                String article = "";

                if (entry.charAt(0) == 'a' || entry.charAt(0) == 'e' || entry.charAt(0) == 'i'
                        || entry.charAt(0) == 'o' || entry.charAt(0) == 'u') {
                    article = "An";
                } else {
                    article = "A";
                }

                if (response.isSuccessful()) {
                    Toast.makeText(getContext(),
                            (article + " " + entry + " is plugged into the server"), Toast.LENGTH_SHORT).show();
                }
            }
        });

        progressBar.hide();
        currentDevice.setText(entry);
    }

    /**
     * retrieves array of all Labels stored in database
     */
    public void fillLabelArray() {

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call<List<String>> call = service.getAllLabels();
        call.enqueue(new Callback<List<String>>() {

            @Override
            public void onFailure(Call<List<String>> call, Throwable t) {

                Utilities.errorToast(getString(R.string.error_no_connection), getContext());
                Utilities.errorToast(getString(R.string.error_no_send), getContext());
                return;

            }

            @Override
            public void onResponse(Call<List<String>> call, Response<List<String>> response) {
                List<String> list = response.body();
                for (String item : list) {

                    System.out.println(item);
                    autoList.add(item);
                }
            }
        });
    }
}