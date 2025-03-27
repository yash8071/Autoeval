## 📌 Steps to Follow

1. **Submit your assignment on Microsoft Teams.**

2. After submission, go to **OneDrive** and download your submitted folder, as shown in the recording.

3. Place the downloaded folder into the `data` folder of the **Autoval** project folder *locally*.

4. Open a terminal or command prompt in the **Autoval** project directory and run the following two commands:

    ```bash
    python main.py -S extract
    python main.py -S code -C
    ```

5. If there are any errors during extraction, they will appear after running these commands.

    - Fix the errors by editing the `.py` file at:

      ```
      data → raw → files
      ```

    - Then, re-run the second command:

      ```bash
      python main.py -S code -C
      ```

    - Make sure to also correct your final `.py` file that you will submit.

6. If everything works correctly, you will find **four generated files** inside the `code` folder.

7. Check the folder path:

    ```
    data → raw → files
    ```

    It should contain:
    - Your `.py` file
    - The four folders included in your submitted zip

8. Open each of the four folders and ensure **all required files are present and accessible**.
