# @title Visualization and Other setup
# Thanks to Joseph Bloom for Tutorial 2.0 where these functions are taken from
# from IPython.display import IFrame
# from transformer_lens.utils import test_prompt
# import os


html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

def get_dashboard_html(model_name, neuronpedia_id, feature_idx, html_template = html_template):
    return html_template.format(model_name, neuronpedia_id, feature_idx)


def generate_combined_dashboard_html(model_name, neuronpedia_id, clusters, feature_indices, output_file="combined_dashboard.html", html_template=html_template):

    #html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    # Start the HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Combined Feature Dashboards</title>
    </head>
    <body>
        <h1>Combined Feature Dashboards</h1>
    """

    # Add each iframe separated by a line break
    for feature_idx in feature_indices:
        dashboard_url = html_template.format(model_name, neuronpedia_id, feature_idx)
        iframe_html = f"""
        <div style="margin-bottom: 50px;">
            <h2>Feature {feature_idx}</h2>
            <h3>Cluster {clusters[layer_of_interest][feature_idx]}</h3>
            <iframe src="{dashboard_url}" width="1000" height="300" frameborder="0">
                Your browser does not support iframes.
            </iframe>
        </div>
        """
        html_content += iframe_html

    # Close the HTML tags
    html_content += """
    </body>
    </html>
    """

    # Write the HTML content to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Combined dashboard saved to {output_file}")