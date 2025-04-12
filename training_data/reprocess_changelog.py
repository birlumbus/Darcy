
'''
Removes data of specified labels from a copied version of dataset_1_changelog.txt
In this case, dataset_2_changelog needed removal of:
    <others-impressions-of-darcy>
    <darcy-actions>
In addition, '<darcy-dialogue>' was changed to '<user>'
'''


def delete_impressions_and_actions(changelog_file):
    with open(changelog_file, "r", encoding="utf-8") as changelog:
        lines = changelog.readlines()
    
    filtered_lines = []
    for line in lines: 
        if line.startswith("<others-impressions-of-darcy>") or line.startswith("<darcy-actions>"):
            continue

        if line.startswith("<darcy-dialogue>"):
            filtered_lines.append("<user>\n")

        filtered_lines.append(line)

    with open(changelog_file, "w", encoding="utf-8") as changelog:
        changelog.writelines(filtered_lines)


def main():
    changelog_file = f"./unprocessed_data/processing_changelogs/dataset_2_changelog.txt"
    print("Deleting impressions and actions...")
    delete_impressions_and_actions(changelog_file)
    print("Impressions and actions deleted.\n")
    

if __name__ == "__main__":
    main()
