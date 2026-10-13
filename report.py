"""PDF one-page executive summary. Results come exclusively from executed ML models."""
import io
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
if (FONT_DIR / 'DejaVuSans.ttf').exists():
    pdfmetrics.registerFont(TTFont('DVS', str(FONT_DIR / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DVS-Bold', str(FONT_DIR / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DVS', normal='DVS', bold='DVS-Bold', italic='DVS', boldItalic='DVS-Bold')
    MAIN_FONT, BOLD_FONT = 'DVS', 'DVS-Bold'
else:
    MAIN_FONT, BOLD_FONT = 'Helvetica', 'Helvetica-Bold'

TEAL = colors.HexColor('#087f80')
INK = colors.HexColor('#153042')


def create_onepager(model_name=None, results=None, n=10000, positive_rate=None, threshold=.5,
                    top_variables=None, custom_dataset=False):
    """Return PDF bytes; leave explicitly unmeasured entries when no results were supplied."""
    data = io.BytesIO()
    pdf = SimpleDocTemplate(data, pagesize=(21*cm, 29.7*cm), leftMargin=1.65*cm,
                            rightMargin=1.65*cm, topMargin=1.35*cm, bottomMargin=1.3*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainX', parent=styles['Title'], fontName=BOLD_FONT,
                              fontSize=19, leading=23, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle(name='SectionX', parent=styles['Heading2'], fontName=BOLD_FONT,
                              fontSize=10.2, leading=13, textColor=TEAL, spaceBefore=11, spaceAfter=4))
    styles.add(ParagraphStyle(name='BodyX', parent=styles['BodyText'], fontName=MAIN_FONT,
                              fontSize=8.6, leading=12.7, textColor=INK, spaceAfter=4))
    styles.add(ParagraphStyle(name='SmallX', parent=styles['BodyX'], fontSize=7.5, leading=10))
    story = [Paragraph('EXPLAINMAINT | Classification explicable', styles['MainX']),
             Paragraph('ENSI - Data Mining - Challenge 1 - Maintenance predictive industrielle (cas 15)', styles['SmallX']),
             HRFlowable(width='100%', thickness=1.4, color=TEAL, spaceBefore=6, spaceAfter=7),
             Paragraph('1  |  Problematique et donnees', styles['SectionX'])]
    if custom_dataset:
        dataset_txt = 'Jeu CSV importe, cible binaire selectionnee dans l interface.'
    else:
        dataset_txt = ('AI4I 2020 (UCI, ID 601), jeu synthetique. 10 000 observations a l origine. '
                       'Cible : panne machine (1) contre fonctionnement (0). Six facteurs : type, temperatures, '
                       'vitesse, couple et usure. TWF/HDF/PWF/OSF/RNF et identifiants exclus pour eviter la fuite de cible.')
    story.append(Paragraph(dataset_txt, styles['BodyX']))
    rate = f'{positive_rate:.2%}' if positive_rate is not None else 'a mesurer a l execution'
    story.append(Paragraph(f'Observations utilisees : <b>{n}</b> | Taux de classe positive : <b>{rate}</b>. '
                           'Decoupage aleatoire stratifie, graine 42, preprocessing ajuste sur apprentissage seul.', styles['BodyX']))
    story.append(Paragraph('2  |  Modeles, evaluation et choix', styles['SectionX']))
    story.append(Paragraph('Regression logistique, arbre, KNN, Random Forest, XGBoost. AUC-ROC, AP (PR-AUC), '
                           'precision, rappel et F1 ; choix du seuil selon le cout des pannes non detectees et des fausses alertes.', styles['BodyX']))
    if results:
        table_data = [['Modèle retenu', str(model_name)], ['AUC-ROC', str(results['AUC-ROC'])],
                      ['PR-AUC (AP)', str(results['PR-AUC (AP)'])], ['Précision (panne)', str(results['Précision'])],
                      ['Rappel (panne)', str(results['Rappel'])], ['F1 (panne)', str(results['F1'])],
                      ['Seuil de décision', f'{threshold:.2f}']]
    else:
        table_data = [['Statut des mesures', 'Non exécutées dans cette livraison statique'],
                      ['AUC, AP, P, R, F1', 'À renseigner après entraînement dans l’application'],
                      ['Seuil de décision', 'À définir dans le module Évaluation']]
    cell_style = ParagraphStyle(name='TableX', parent=styles['BodyX'], fontSize=7.5, leading=10, spaceAfter=0)
    table_data = [[Paragraph(str(a), cell_style), Paragraph(str(b), cell_style)] for a, b in table_data]
    tbl = Table(table_data, colWidths=[6.4*cm, 10.85*cm], hAlign='LEFT')
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#E8F5F4')),
                             ('TEXTCOLOR',(0,0),(-1,-1),INK), ('FONTNAME',(0,0),(0,-1),BOLD_FONT),
                             ('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),5),
                             ('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-2),.35,colors.HexColor('#C9DBDD'))]))
    story.append(tbl)
    story.append(Paragraph('3  |  Explicabilite et actions', styles['SectionX']))
    if top_variables:
        vars_txt = ', '.join(str(x).replace('num__', '').replace('cat__', '') for x in top_variables[:4])
        story.append(Paragraph(f'Principales variables SHAP dans l echantillon explique : <b>{vars_txt}</b>. '
                               'SHAP estime le sens et l amplitude des contributions ; LIME approxime localement '
                               'le comportement du meme modele. Verifier leur convergence individu par individu.', styles['BodyX']))
    else:
        story.append(Paragraph('Examiner les importances natives et par permutation, le SHAP beeswarm / dependance, '
                               'puis SHAP waterfall et LIME sur le meme individu. Les facteurs dominants sont a relever '
                               'sur la sortie calculee ; aucune importance numerique n est supposee avant execution.', styles['BodyX']))
    story.append(Paragraph('Decision : prioriser les inspections des cas classes a fort risque et controler physiquement '
                           'l usure, le couple, la rotation et l ecart thermique ; fixer le seuil en concertation avec la maintenance.', styles['BodyX']))
    story.append(Paragraph('4  |  Analyse critique et limites', styles['SectionX']))
    story.append(Paragraph('SHAP et LIME decrivent le modele, pas une causalite ni un diagnostic mecanique certain. '
                           'Confronter les signes aux connaissances terrain et documenter leurs divergences. '
                           'Les variables UCI retenues ne comportent pas de donnees personnelles directement sensibles ; '
                           'un CSV externe exige une analyse d equite (age, sexe, etc.). XGBoost peut mieux capter '
                           'des interactions ; une regression logistique reste plus directement lisible : recommander '
                           'selon AP, rappel et contraintes de gouvernance, pas sur le nom de l algorithme. '
                           'Le dataset est synthetique : pas de validation chronologique ni de preuve de performance en production.', styles['BodyX']))
    story.append(Spacer(1,7))
    story.append(HRFlowable(width='100%', thickness=.7, color=TEAL))
    story.append(Paragraph('Source : UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset, '
                           'DOI 10.24432/C5HS5C (CC BY 4.0). Resultats chiffres uniquement si issus de l execution.', styles['SmallX']))
    pdf.build(story)
    return data.getvalue()

if __name__ == '__main__':
    from pathlib import Path
    Path('one_pager_modele.pdf').write_bytes(create_onepager())
