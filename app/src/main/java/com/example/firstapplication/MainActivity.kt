package com.example.firstapplication

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.firstapplication.databinding.ActivityMainBinding
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    external fun stringFromJNI(): String

    companion object {
        init {
            System.loadLibrary("firstapplication")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.sampleText.text = "Loading model..."

        thread {
            val result = stringFromJNI()

            runOnUiThread {
                binding.sampleText.text = result
            }
        }
    }
}