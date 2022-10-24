
**TAMSAT-ALERT Python Tool**

Written by: *Vicky Boult*

Institution: *University of Reading*

Date: *24th October 2022*

The TAMSAT-ALERT Python Tool is a piece of software (Python code) designed to allow users to operationally generate TAMSAT-ALERT soil moisture forecasts for for their region (within Africa) and season of interest. The software is designed so that users only need to type one line of code and the software does all the rest!
Users specify the period and region of interest, and the meteorological tercile forecast, and the Python Tool will download the required data, apply the weighting, and output the forecast in a number of formats.
The TAMSAT team intend to further develop the tool in future to improve functionality and fix any bugs, but by providing the software in this way, it will be easy for users to keep up-to-date with the latest version whilst continuing to operationally produce soil moisture forecasts.

**Example TAMSAT-ALERT forecasts for Kenya's 2021 MAM season:**

![image](https://user-images.githubusercontent.com/29065315/116704642-8e891f00-a9c3-11eb-8c38-c35ec9472356.png)

**Getting started:**

The TAMSAT-ALERT Python Tool and underlying data are free to use. 

1) Extract T-A_API.zip

Begin by downloading and extracting the T-A_API.zip folder. Within this folder, you will find the Python code and directories required to generate a TAMSAT-ALERT forecast. Do not move or delete any files or folders as the system relies on the correct arrangement of these to store data and outputs. 

2) Reproducing Testcases

To use the TAMSAT-ALERT Python Tool, we recommend first following instructions in **'Reproduce_Testcases.pdf'**. This will ensure that your system is set-up and operating correctly before generating your own bespoke forecasts. It also helps the TAMSAT team to troubleshoot any errors you may have. 

3) Bespoke Forecasts

Once you have successfully reproduced the testcases, you can move on to generate forecasts for your own region and season of interest. Follow instructions in **'Bespoke_Forecasting.pdf'**.

**Operational Configurations:**

Alongside the main Python script to generate TAMSAT-ALERT forecasts are a series of operational configurations. These are 'add-ons' increasing the functionality of the TAMSAT-ALERT Python Tool to complete commonly requested tasks, such as adding local geographic boundaries. Check out **'Using_Operational_Configurations.pdf'** for information on how to use them. Further operational configurations will be added over time as required.

**Troubleshooting:**

If you experience any problems in using the TAMSAT-ALERT Python Tool, we first encourage you to ensure you have successfully reproduced the testcases (above). The tool is written in Python, and so standard Python error messages may be addressed through a simple Google search. If your problem persists, please contact Vicky Boult at the following email address: v(dot)l(dot)boult(at)reading(dot)ac(dot)uk.

