## 📌 Steps to Follow

1. **Submit your assignment on Microsoft Teams.**

2. After submission, go to **OneDrive** and download your submitted folder, as shown in the recording.

3. Place the downloaded folder into the `data` folder of the **`Autoval`** project folder **locally**.

4. Open a terminal or command prompt in the `Autoval` project directory and run the following two commands:

    ```bash
    python main.py -S extract
    python main.py -S code -C
    ```

5. If there are any errors during extraction, you will see them after running these commands.

    - To solve the errors, make corrections in the `.py` file available at:

      ```
      data → raw → files
      ```

    - Then **run the second command again**:

      ```bash
      python main.py -S code -C
      ```

    - Also, make sure to make the same correction in your final `.py` file, which you will submit.

6. If everything works correctly, you will find **four generated files** inside the `code` folder.

7. Also, check the folder path:

    ```
    data → raw → files
    ```

    This should contain:

    - Your `.py` file  
    - The four folders you included in your submitted zip

8. Open each of the four folders and make sure **all required files are there and accessible**.
