VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} frmExportCSVVerloning 
   Caption         =   "Selecteer Overzicht en Exportcriteria"
   ClientHeight    =   2370
   ClientLeft      =   45
   ClientTop       =   330
   ClientWidth     =   9735
   OleObjectBlob   =   "frmExportCSVVerloning.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "frmExportCSVVerloning"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Option Explicit
Private Sub cmdAnnuleren_Click()
    glAnnuleren = True
    glOK = False
    Me.Hide
End Sub
Private Sub cmdSelecteerCSV_Click()
Dim strBewaarlocatie As String
    strBewaarlocatie = tbCSVBestand.value
    With Application.FileDialog(msoFileDialogOpen)
        .AllowMultiSelect = False
        .Filters.Clear
        .Filters.Add "Excel bestand", "*.csv"
        .FilterIndex = 1
        .InitialFileName = IIf(FileExists(strBewaarlocatie), strBewaarlocatie, "")

         If .Show = -1 Then ' Er is een bestand gekozen
            tbCSVBestand.value = .SelectedItems(1)
         Else
            tbCSVBestand.value = strBewaarlocatie
         End If
     End With
End Sub
Private Sub cmdOK_Click()
    If Not WsExists(cmbVerloonOverzicht.value, ThisWorkbook) Then
        MsgBox "Selecteer een bestaand VerloonOverzicht !"
        cmbVerloonOverzicht.SetFocus
        Exit Sub
    End If
    If Not FileExists(tbCSVBestand.value) Then
       MsgBox "Selecteer een bestaand CSV bestand !"
       tbCSVBestand.SetFocus
       Exit Sub
    End If
    
    glOK = True
    glAnnuleren = False
    
    Me.Hide
End Sub
Private Sub VulVerloonOverzicht()
Dim ws As Worksheet
cmbVerloonOverzicht.Clear
For Each ws In ThisWorkbook.Worksheets
    If Left(UCase(ws.Name) & String(5, " "), 5) = "VERL_" Then
        cmbVerloonOverzicht.AddItem ws.Name
    End If
Next ws
End Sub
Private Sub UserForm_Activate()
'   Default is het verloonoverzicht met de laagste index
'   Eerst besturingselementen goed zetten.
    
    Application.EnableEvents = False
    VulVerloonOverzicht
    cmbVerloonOverzicht.SetFocus
    If cmbVerloonOverzicht.ListCount > 0 Then cmbVerloonOverzicht.ListIndex = 0
    Application.EnableEvents = True
End Sub
