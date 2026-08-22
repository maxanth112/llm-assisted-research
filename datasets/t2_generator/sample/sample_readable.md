# T2 Diagnostic Test Items - Sample Dataset

Generated 16 items across 4 regimes.

## CLEAN Regime

Evidence uniquely identifies one hypothesis. Multiple evidence items all point to the guilty suspect.

---

### Item 1: t2_clean_theft_000

**Template:** theft

**Narrative:**

At Manufacturing Solutions Inc, a valuable prototype device was stolen from Warehouse District on March 15, 2024 between 8:00 PM and 11:00 PM. The security investigation has narrowed down to 4 employees who had access during that timeframe.

**Question:**

Based on the available evidence, who is most likely responsible for the theft?

**Hypotheses:**

1. Grey Martinez is responsible ✓
2. Alex Chen is responsible
3. Ellis Kim is responsible
4. Harper Quinn is responsible

**Evidence (6 items):**

- **E001** [high]: Security camera footage shows Grey Martinez entering Warehouse District at 8:15 PM carrying an empty bag and exiting at 8:45 PM with a bulging bag.
  - *Supports:* Grey Martinez
- **E002** [high]: Grey Martinez's keycard was used to unlock the secure storage area within Warehouse District at 8:15 PM, which contained the stolen prototype device.
  - *Supports:* Grey Martinez
- **E003** [high]: The stolen prototype device was found hidden in Grey Martinez's personal locker during the investigation, wrapped in Grey Martinez's distinctive jacket.
  - *Supports:* Grey Martinez
- **E004** [high]: Alex Chen was confirmed to be attending a company training session in Building D - Storage Facility during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Alex Chen
- **E005** [high]: Ellis Kim was confirmed to be attending a company training session in Building C - Server Room during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Ellis Kim
- **E006** [high]: Harper Quinn was confirmed to be attending a company training session in Downtown Branch Office during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Harper Quinn

**Gold Answer:**

Grey Martinez is responsible

**Gold Reasoning:**

Multiple independent pieces of high-quality diagnostic evidence (access logs, physical evidence, witness testimony, and forensic analysis) all consistently point to Grey Martinez. All other suspects have verified alibis placing them elsewhere during the incident.

---

### Item 2: t2_clean_sabotage_001

**Template:** sabotage

**Narrative:**

At TechCorp Industries's Main Office - 3rd Floor, critical equipment was sabotaged on March 15, 2024, causing a production shutdown. Internal investigation identified 4 employees with the technical knowledge and access to perform the sabotage.

**Question:**

Based on the available evidence, who is most likely responsible for the sabotage?

**Hypotheses:**

1. Indigo Taylor is responsible ✓
2. Dana Patel is responsible
3. Alex Chen is responsible
4. Ellis Kim is responsible

**Evidence (6 items):**

- **E001** [high]: System logs show Indigo Taylor's credentials were used to modify critical safety settings on the equipment at 8:15 PM, immediately before the malfunction occurred.
  - *Supports:* Indigo Taylor
- **E002** [high]: Custodian Mike Thompson observed Indigo Taylor working alone on the equipment at 8:15 PM and noted that Indigo Taylor appeared nervous when questioned.
  - *Supports:* Indigo Taylor
- **E003** [high]: Specialized calibration tools registered to Indigo Taylor were found at the scene with fresh fingerprints matching Indigo Taylor.
  - *Supports:* Indigo Taylor
- **E004** [high]: Dana Patel was confirmed to be attending a company training session in Building B - Research Lab during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Dana Patel
- **E005** [high]: Alex Chen was confirmed to be attending a company training session in Building B - Research Lab during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Alex Chen
- **E006** [high]: Ellis Kim was confirmed to be attending a company training session in Building C - Server Room during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Ellis Kim

**Gold Answer:**

Indigo Taylor is responsible

**Gold Reasoning:**

Multiple independent pieces of high-quality diagnostic evidence (access logs, physical evidence, witness testimony, and forensic analysis) all consistently point to Indigo Taylor. All other suspects have verified alibis placing them elsewhere during the incident.

---

### Item 3: t2_clean_data_breach_002

**Template:** data_breach

**Narrative:**

At Manufacturing Solutions Inc, unauthorized access to sensitive customer data was detected on April 22, 2024. The breach occurred between 6:30 PM and 9:30 PM. Security analysis identified 4 employees with sufficient system privileges to access the compromised database.

**Question:**

Based on the available evidence, who is most likely responsible for the data breach?

**Hypotheses:**

1. Jordan Lee is responsible ✓
2. Indigo Taylor is responsible
3. Dana Patel is responsible
4. Grey Martinez is responsible

**Evidence (6 items):**

- **E001** [high]: Network logs show 15 GB of customer data was exfiltrated from the IP address assigned to Jordan Lee's workstation at 8:15 PM.
  - *Supports:* Jordan Lee
- **E002** [high]: Database audit logs record Jordan Lee's credentials accessing sensitive customer tables at 8:15 PM, with queries specifically extracting personal information.
  - *Supports:* Jordan Lee
- **E003** [high]: An unauthorized encrypted USB device was detected on Jordan Lee's computer at 8:15 PM, and the device was found in Jordan Lee's desk drawer.
  - *Supports:* Jordan Lee
- **E004** [high]: Indigo Taylor was confirmed to be attending a company training session in Building C - Server Room during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Indigo Taylor
- **E005** [high]: Dana Patel was confirmed to be attending a company training session in Research Campus - North Wing during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Dana Patel
- **E006** [high]: Grey Martinez was confirmed to be attending a company training session in Warehouse District during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Grey Martinez

**Gold Answer:**

Jordan Lee is responsible

**Gold Reasoning:**

Multiple independent pieces of high-quality diagnostic evidence (access logs, physical evidence, witness testimony, and forensic analysis) all consistently point to Jordan Lee. All other suspects have verified alibis placing them elsewhere during the incident.

---

### Item 4: t2_clean_contamination_003

**Template:** contamination

**Narrative:**

At Global Logistics Co's Building C - Server Room, a batch of products was contaminated on March 15, 2024, requiring a costly recall. Quality control investigation narrowed the incident to 4 staff members who handled the batch during production.

**Question:**

Based on the available evidence, who is most likely responsible for the contamination?

**Hypotheses:**

1. Grey Martinez is responsible ✓
2. Finley Brooks is responsible
3. Harper Quinn is responsible
4. Dana Patel is responsible

**Evidence (6 items):**

- **E001** [high]: Production records show Grey Martinez was the sole operator assigned to the contaminated batch during the night shift when contamination occurred.
  - *Supports:* Grey Martinez
- **E002** [high]: Equipment assigned to Grey Martinez tested positive for the specific contaminant substance, and Grey Martinez's gloves also showed traces of the same substance.
  - *Supports:* Grey Martinez
- **E003** [high]: Reception Staff Chris Anderson documented that Grey Martinez failed to follow the required sterilization protocol during the shift, which would have prevented the contamination.
  - *Supports:* Grey Martinez
- **E004** [high]: Finley Brooks was confirmed to be attending a company training session in Research Campus - North Wing during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Finley Brooks
- **E005** [high]: Harper Quinn was confirmed to be attending a company training session in Downtown Branch Office during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Harper Quinn
- **E006** [high]: Dana Patel was confirmed to be attending a company training session in Downtown Branch Office during the incident, verified by sign-in records and instructor confirmation.
  - *Contradicts:* Dana Patel

**Gold Answer:**

Grey Martinez is responsible

**Gold Reasoning:**

Multiple independent pieces of high-quality diagnostic evidence (access logs, physical evidence, witness testimony, and forensic analysis) all consistently point to Grey Martinez. All other suspects have verified alibis placing them elsewhere during the incident.

---


## DECOY Regime

Clean evidence + salient-but-non-diagnostic decoy evidence pointing at innocent suspects.

---

### Item 1: t2_decoy_theft_004

**Template:** theft

**Narrative:**

At DataVault Systems, a valuable server access key was stolen from Warehouse District on March 15, 2024 between 8:00 PM and 10:00 PM. The security investigation has narrowed down to 4 employees who had access during that timeframe.

**Question:**

Based on the available evidence, who is most likely responsible for the theft?

**Hypotheses:**

1. Grey Martinez is responsible ✓
2. Dana Patel is responsible
3. Indigo Taylor is responsible
4. Finley Brooks is responsible

**Evidence (7 items):**

- **E001** [high]: Security camera footage shows Grey Martinez entering Warehouse District at 8:15 PM carrying an empty bag and exiting at 8:45 PM with a bulging bag.
  - *Supports:* Grey Martinez
- **E002** [high]: The stolen prototype device was found hidden in Grey Martinez's personal locker during the investigation, wrapped in Grey Martinez's distinctive jacket.
  - *Supports:* Grey Martinez
- **E003** [low]: Dana Patel had recently expressed frustration about not being assigned to work on the prototype project, stating 'I should have been chosen for that team.'
  - *Supports:* Dana Patel
- **E004** [low]: Indigo Taylor was seen browsing competitor company websites during work hours the week before the theft.
  - *Supports:* Indigo Taylor
- **E005** [high]: Dana Patel was confirmed to be in Building A - East Wing during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Dana Patel
- **E006** [high]: Indigo Taylor was confirmed to be in Main Office - 3rd Floor during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Indigo Taylor
- **E007** [high]: Finley Brooks was confirmed to be in Main Office - 3rd Floor during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Finley Brooks

**Gold Answer:**

Grey Martinez is responsible

**Gold Reasoning:**

Despite salient decoy evidence suggesting motive or opportunity for other suspects, high-quality diagnostic evidence (access logs, physical evidence) definitively points to Grey Martinez. The decoy evidence for other suspects is circumstantial and non-diagnostic, and all innocent suspects have verified alibis.

---

### Item 2: t2_decoy_sabotage_005

**Template:** sabotage

**Narrative:**

At DataVault Systems's Building C - Server Room, critical equipment was sabotaged on June 12, 2024, causing a production shutdown. Internal investigation identified 4 employees with the technical knowledge and access to perform the sabotage.

**Question:**

Based on the available evidence, who is most likely responsible for the sabotage?

**Hypotheses:**

1. Jordan Lee is responsible ✓
2. Grey Martinez is responsible
3. Indigo Taylor is responsible
4. Finley Brooks is responsible

**Evidence (7 items):**

- **E001** [high]: System logs show Jordan Lee's credentials were used to modify critical safety settings on the equipment at 8:15 PM, immediately before the malfunction occurred.
  - *Supports:* Jordan Lee
- **E002** [high]: Specialized calibration tools registered to Jordan Lee were found at the scene with fresh fingerprints matching Jordan Lee.
  - *Supports:* Jordan Lee
- **E003** [low]: Grey Martinez was overheard complaining about management decisions regarding equipment maintenance schedules two weeks before the incident.
  - *Supports:* Grey Martinez
- **E004** [low]: Indigo Taylor has extensive technical knowledge of the sabotaged equipment type and previously worked as a systems engineer at a competitor.
  - *Supports:* Indigo Taylor
- **E005** [high]: Grey Martinez was confirmed to be in Building B - Research Lab during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Grey Martinez
- **E006** [high]: Indigo Taylor was confirmed to be in Building A - East Wing during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Indigo Taylor
- **E007** [high]: Finley Brooks was confirmed to be in Warehouse District during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Finley Brooks

**Gold Answer:**

Jordan Lee is responsible

**Gold Reasoning:**

Despite salient decoy evidence suggesting motive or opportunity for other suspects, high-quality diagnostic evidence (access logs, physical evidence) definitively points to Jordan Lee. The decoy evidence for other suspects is circumstantial and non-diagnostic, and all innocent suspects have verified alibis.

---

### Item 3: t2_decoy_data_breach_006

**Template:** data_breach

**Narrative:**

At DataVault Systems, unauthorized access to sensitive customer data was detected on May 8, 2024. The breach occurred between 8:00 PM and 11:00 PM. Security analysis identified 4 employees with sufficient system privileges to access the compromised database.

**Question:**

Based on the available evidence, who is most likely responsible for the data breach?

**Hypotheses:**

1. Indigo Taylor is responsible ✓
2. Ellis Kim is responsible
3. Dana Patel is responsible
4. Jordan Lee is responsible

**Evidence (7 items):**

- **E001** [high]: Network logs show 15 GB of customer data was exfiltrated from the IP address assigned to Indigo Taylor's workstation at 8:15 PM.
  - *Supports:* Indigo Taylor
- **E002** [high]: An unauthorized encrypted USB device was detected on Indigo Taylor's computer at 8:15 PM, and the device was found in Indigo Taylor's desk drawer.
  - *Supports:* Indigo Taylor
- **E003** [low]: Ellis Kim recently updated their LinkedIn profile and has been in contact with recruiters from competing firms.
  - *Supports:* Ellis Kim
- **E004** [low]: Dana Patel was observed working unusually late hours in the week preceding the breach, though no specific suspicious activity was noted.
  - *Supports:* Dana Patel
- **E005** [high]: Ellis Kim was confirmed to be in Warehouse District during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Ellis Kim
- **E006** [high]: Dana Patel was confirmed to be in Building B - Research Lab during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Dana Patel
- **E007** [high]: Jordan Lee was confirmed to be in Building A - East Wing during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Jordan Lee

**Gold Answer:**

Indigo Taylor is responsible

**Gold Reasoning:**

Despite salient decoy evidence suggesting motive or opportunity for other suspects, high-quality diagnostic evidence (access logs, physical evidence) definitively points to Indigo Taylor. The decoy evidence for other suspects is circumstantial and non-diagnostic, and all innocent suspects have verified alibis.

---

### Item 4: t2_decoy_contamination_007

**Template:** contamination

**Narrative:**

At DataVault Systems's Building A - East Wing, a batch of products was contaminated on April 22, 2024, requiring a costly recall. Quality control investigation narrowed the incident to 4 staff members who handled the batch during production.

**Question:**

Based on the available evidence, who is most likely responsible for the contamination?

**Hypotheses:**

1. Casey Morgan is responsible ✓
2. Ellis Kim is responsible
3. Harper Quinn is responsible
4. Grey Martinez is responsible

**Evidence (7 items):**

- **E001** [high]: Production records show Casey Morgan was the sole operator assigned to the contaminated batch during the night shift when contamination occurred.
  - *Supports:* Casey Morgan
- **E002** [high]: Equipment assigned to Casey Morgan tested positive for the specific contaminant substance, and Casey Morgan's gloves also showed traces of the same substance.
  - *Supports:* Casey Morgan
- **E003** [low]: Ellis Kim had received a written warning three months ago for a minor quality control violation in an unrelated production area.
  - *Supports:* Ellis Kim
- **E004** [low]: Harper Quinn was seen entering the facility early on the day of contamination, though their shift didn't start until later.
  - *Supports:* Harper Quinn
- **E005** [high]: Ellis Kim was confirmed to be in Research Campus - North Wing during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Ellis Kim
- **E006** [high]: Harper Quinn was confirmed to be in Main Office - 3rd Floor during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Harper Quinn
- **E007** [high]: Grey Martinez was confirmed to be in Main Office - 3rd Floor during the critical incident window, verified by timestamped security badge records.
  - *Contradicts:* Grey Martinez

**Gold Answer:**

Casey Morgan is responsible

**Gold Reasoning:**

Despite salient decoy evidence suggesting motive or opportunity for other suspects, high-quality diagnostic evidence (access logs, physical evidence) definitively points to Casey Morgan. The decoy evidence for other suspects is circumstantial and non-diagnostic, and all innocent suspects have verified alibis.

---


## CONFLICT Regime

Sources genuinely disagree. Source-precedence rule determines the gold answer.

---

### Item 1: t2_conflict_theft_008

**Template:** theft

**Narrative:**

At Global Logistics Co, a valuable confidential research document was stolen from Building A - East Wing on May 8, 2024 between 8:00 PM and 10:00 PM. The security investigation has narrowed down to 4 employees who had access during that timeframe.

**Question:**

Based on the available evidence, who is most likely responsible for the theft?

**Hypotheses:**

1. Dana Patel is responsible ✓
2. Alex Chen is responsible
3. Ellis Kim is responsible
4. Harper Quinn is responsible

**Evidence (6 items):**

- **E001** [high]: Security system logs show Dana Patel's keycard was used to access Building A - East Wing at 8:15 PM, the exact time when the theft occurred.
  - *Supports:* Dana Patel
  - *Contradicts:* Alex Chen
- **E002** [high]: Digital access logs indicate Dana Patel's credentials unlocked the secure storage containing the stolen item at 8:15 PM.
  - *Supports:* Dana Patel
  - *Contradicts:* Alex Chen
- **E003** [high]: Maintenance Lead Sam Rodriguez testified that they clearly saw Alex Chen leaving Building A - East Wing at 8:45 PM carrying a bag matching the description of the stolen item's container.
  - *Supports:* Alex Chen
  - *Contradicts:* Dana Patel
- **E004** [high]: Security Officer Sarah Johnson provided a statement saying they observed Alex Chen acting suspiciously near Building A - East Wing around 8:15 PM and specifically remembers Alex Chen looking around nervously.
  - *Supports:* Alex Chen
  - *Contradicts:* Dana Patel
- **E005** [medium]: Ellis Kim was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Ellis Kim
- **E006** [medium]: Harper Quinn was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Harper Quinn

**Source Precedence Rule:**

When evidence conflicts, official system logs and forensic records take precedence over witness testimony and circumstantial evidence.

**Gold Answer:**

Dana Patel is responsible

**Gold Reasoning:**

The evidence presents a genuine conflict: official system logs and forensic records point to Dana Patel, while witness testimony points to Alex Chen. Applying the source precedence rule (official records take precedence over witness testimony), Dana Patel is determined to be responsible. The official digital records are considered more reliable than human observation, which can be subject to error or misidentification.

---

### Item 2: t2_conflict_sabotage_009

**Template:** sabotage

**Narrative:**

At BioResearch Labs's Building C - Server Room, critical equipment was sabotaged on March 15, 2024, causing a production shutdown. Internal investigation identified 4 employees with the technical knowledge and access to perform the sabotage.

**Question:**

Based on the available evidence, who is most likely responsible for the sabotage?

**Hypotheses:**

1. Blake Rivera is responsible ✓
2. Finley Brooks is responsible
3. Jordan Lee is responsible
4. Harper Quinn is responsible

**Evidence (6 items):**

- **E001** [high]: System audit logs show Blake Rivera's network credentials were used to modify the equipment control settings at 8:15 PM, causing the malfunction.
  - *Supports:* Blake Rivera
  - *Contradicts:* Finley Brooks
- **E002** [high]: Automated monitoring systems recorded login from Blake Rivera's account executing unauthorized configuration changes at 8:15 PM.
  - *Supports:* Blake Rivera
  - *Contradicts:* Finley Brooks
- **E003** [high]: IT Administrator Pat Williams testified that they saw Finley Brooks physically working on the equipment at 8:15 PM with tools, and no one else was present in the area.
  - *Supports:* Finley Brooks
  - *Contradicts:* Blake Rivera
- **E004** [high]: Maintenance Lead Sam Rodriguez reported that Finley Brooks was the only person with the technical expertise present during the sabotage window and specifically saw Finley Brooks accessing the equipment control panel.
  - *Supports:* Finley Brooks
  - *Contradicts:* Blake Rivera
- **E005** [medium]: Jordan Lee was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Jordan Lee
- **E006** [medium]: Harper Quinn was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Harper Quinn

**Source Precedence Rule:**

When evidence conflicts, official system logs and forensic records take precedence over witness testimony and circumstantial evidence.

**Gold Answer:**

Blake Rivera is responsible

**Gold Reasoning:**

The evidence presents a genuine conflict: official system logs and forensic records point to Blake Rivera, while witness testimony points to Finley Brooks. Applying the source precedence rule (official records take precedence over witness testimony), Blake Rivera is determined to be responsible. The official digital records are considered more reliable than human observation, which can be subject to error or misidentification.

---

### Item 3: t2_conflict_data_breach_010

**Template:** data_breach

**Narrative:**

At DataVault Systems, unauthorized access to sensitive customer data was detected on June 12, 2024. The breach occurred between 7:00 PM and 10:00 PM. Security analysis identified 4 employees with sufficient system privileges to access the compromised database.

**Question:**

Based on the available evidence, who is most likely responsible for the data breach?

**Hypotheses:**

1. Blake Rivera is responsible ✓
2. Harper Quinn is responsible
3. Alex Chen is responsible
4. Indigo Taylor is responsible

**Evidence (6 items):**

- **E001** [high]: Network forensic logs show data exfiltration originated from IP address 192.168.1.42, which is assigned to Blake Rivera's workstation.
  - *Supports:* Blake Rivera
  - *Contradicts:* Harper Quinn
- **E002** [high]: Database access logs record Blake Rivera's credentials querying and extracting sensitive customer records at 8:15 PM.
  - *Supports:* Blake Rivera
  - *Contradicts:* Harper Quinn
- **E003** [high]: Security Officer Sarah Johnson stated they saw Harper Quinn working alone in the server room at 8:15 PM with a laptop and USB drive, which is highly unusual and against protocol.
  - *Supports:* Harper Quinn
  - *Contradicts:* Blake Rivera
- **E004** [high]: Maintenance Lead Sam Rodriguez testified that Harper Quinn asked them detailed questions about database security protocols the week before the breach and seemed overly interested in access logs.
  - *Supports:* Harper Quinn
  - *Contradicts:* Blake Rivera
- **E005** [medium]: Alex Chen was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Alex Chen
- **E006** [medium]: Indigo Taylor was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Indigo Taylor

**Source Precedence Rule:**

When evidence conflicts, official system logs and forensic records take precedence over witness testimony and circumstantial evidence.

**Gold Answer:**

Blake Rivera is responsible

**Gold Reasoning:**

The evidence presents a genuine conflict: official system logs and forensic records point to Blake Rivera, while witness testimony points to Harper Quinn. Applying the source precedence rule (official records take precedence over witness testimony), Blake Rivera is determined to be responsible. The official digital records are considered more reliable than human observation, which can be subject to error or misidentification.

---

### Item 4: t2_conflict_contamination_011

**Template:** contamination

**Narrative:**

At BioResearch Labs's Building B - Research Lab, a batch of products was contaminated on March 15, 2024, requiring a costly recall. Quality control investigation narrowed the incident to 4 staff members who handled the batch during production.

**Question:**

Based on the available evidence, who is most likely responsible for the contamination?

**Hypotheses:**

1. Dana Patel is responsible ✓
2. Indigo Taylor is responsible
3. Finley Brooks is responsible
4. Alex Chen is responsible

**Evidence (6 items):**

- **E001** [high]: Production tracking system shows Dana Patel was the logged operator for the contaminated batch, with their employee ID recorded at each production checkpoint.
  - *Supports:* Dana Patel
  - *Contradicts:* Indigo Taylor
- **E002** [high]: Automated quality control logs indicate Dana Patel's workstation ID processed the contaminated batch during the critical time window.
  - *Supports:* Dana Patel
  - *Contradicts:* Indigo Taylor
- **E003** [high]: Maintenance Lead Sam Rodriguez provided detailed testimony that they personally observed Indigo Taylor handling the batch without proper protective equipment and skipping sterilization steps.
  - *Supports:* Indigo Taylor
  - *Contradicts:* Dana Patel
- **E004** [high]: Reception Staff Chris Anderson stated that Indigo Taylor was the only person they saw working on that production line during the shift, despite what the system logs indicate.
  - *Supports:* Indigo Taylor
  - *Contradicts:* Dana Patel
- **E005** [medium]: Finley Brooks was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Finley Brooks
- **E006** [medium]: Alex Chen was working in a different department during the incident and has no apparent connection to the case.
  - *Contradicts:* Alex Chen

**Source Precedence Rule:**

When evidence conflicts, official system logs and forensic records take precedence over witness testimony and circumstantial evidence.

**Gold Answer:**

Dana Patel is responsible

**Gold Reasoning:**

The evidence presents a genuine conflict: official system logs and forensic records point to Dana Patel, while witness testimony points to Indigo Taylor. Applying the source precedence rule (official records take precedence over witness testimony), Dana Patel is determined to be responsible. The official digital records are considered more reliable than human observation, which can be subject to error or misidentification.

---


## INSUFFICIENT Regime

Evidence is genuinely ambiguous. All suspects have equal evidence.

---

### Item 1: t2_insufficient_theft_012

**Template:** theft

**Narrative:**

At Manufacturing Solutions Inc, a valuable confidential research document was stolen from Warehouse District on April 22, 2024 between 7:00 PM and 9:30 PM. The security investigation has narrowed down to 4 employees who had access during that timeframe.

**Question:**

Based on the available evidence, who is most likely responsible for the theft?

**Hypotheses:**

1. Blake Rivera is responsible
2. Harper Quinn is responsible
3. Casey Morgan is responsible
4. Indigo Taylor is responsible
5. Cannot be determined from available evidence ✓

**Evidence (7 items):**

- **E001** [low]: Badge access logs show all four suspects (Blake Rivera, Harper Quinn, Casey Morgan, Indigo Taylor) accessed Warehouse District at various times during the incident window between 8:15 PM and 9:00 PM.
  - *Supports:* Blake Rivera, Harper Quinn, Casey Morgan, Indigo Taylor
- **E002** [low]: Blake Rivera was observed near Warehouse District around 8:15 PM, which falls within the timeframe when the theft could have occurred.
  - *Supports:* Blake Rivera
- **E003** [low]: Harper Quinn was observed near Warehouse District around 8:30 PM, which falls within the timeframe when the theft could have occurred.
  - *Supports:* Harper Quinn
- **E004** [low]: Casey Morgan was observed near Warehouse District around 8:45 PM, which falls within the timeframe when the theft could have occurred.
  - *Supports:* Casey Morgan
- **E005** [low]: Indigo Taylor was observed near Warehouse District around 9:00 PM, which falls within the timeframe when the theft could have occurred.
  - *Supports:* Indigo Taylor
- **E006** [none]: Security camera footage from the critical period is corrupted and does not show who removed the item from the secure area.
- **E007** [low]: Forensic analysis found multiple sets of fingerprints on the storage cabinet, including partial prints from all four suspects, which is expected given they all work in this area regularly.
  - *Supports:* Blake Rivera, Harper Quinn, Casey Morgan, Indigo Taylor

**Gold Answer:**

Cannot be determined from available evidence

**Gold Reasoning:**

The evidence is genuinely ambiguous and insufficient to identify a specific responsible party. All suspects had equal access, opportunity, and capability. No diagnostic evidence uniquely implicates any individual suspect, and critical forensic data is either unavailable or non-discriminating. A definitive determination cannot be made without additional evidence.

---

### Item 2: t2_insufficient_sabotage_013

**Template:** sabotage

**Narrative:**

At TechCorp Industries's Building A - East Wing, critical equipment was sabotaged on April 22, 2024, causing a production shutdown. Internal investigation identified 4 employees with the technical knowledge and access to perform the sabotage.

**Question:**

Based on the available evidence, who is most likely responsible for the sabotage?

**Hypotheses:**

1. Indigo Taylor is responsible
2. Harper Quinn is responsible
3. Alex Chen is responsible
4. Finley Brooks is responsible
5. Cannot be determined from available evidence ✓

**Evidence (7 items):**

- **E001** [low]: All four suspects (Indigo Taylor, Harper Quinn, Alex Chen, Finley Brooks) have the required technical expertise and training to perform this type of sabotage.
  - *Supports:* Indigo Taylor, Harper Quinn, Alex Chen, Finley Brooks
- **E002** [low]: System logs indicate that all four suspects accessed the equipment control system on the day of the incident, which is routine for their roles.
  - *Supports:* Indigo Taylor, Harper Quinn, Alex Chen, Finley Brooks
- **E003** [low]: Indigo Taylor was working in the area around 8:15 PM and had unsupervised access to the equipment.
  - *Supports:* Indigo Taylor
- **E004** [low]: Harper Quinn was working in the area around 8:30 PM and had unsupervised access to the equipment.
  - *Supports:* Harper Quinn
- **E005** [low]: Alex Chen was working in the area around 8:45 PM and had unsupervised access to the equipment.
  - *Supports:* Alex Chen
- **E006** [low]: Finley Brooks was working in the area around 9:00 PM and had unsupervised access to the equipment.
  - *Supports:* Finley Brooks
- **E007** [low]: The sabotage method used is a standard technique that all four suspects learned in their technical training program.
  - *Supports:* Indigo Taylor, Harper Quinn, Alex Chen, Finley Brooks

**Gold Answer:**

Cannot be determined from available evidence

**Gold Reasoning:**

The evidence is genuinely ambiguous and insufficient to identify a specific responsible party. All suspects had equal access, opportunity, and capability. No diagnostic evidence uniquely implicates any individual suspect, and critical forensic data is either unavailable or non-discriminating. A definitive determination cannot be made without additional evidence.

---

### Item 3: t2_insufficient_data_breach_014

**Template:** data_breach

**Narrative:**

At TechCorp Industries, unauthorized access to sensitive customer data was detected on March 15, 2024. The breach occurred between 6:30 PM and 11:00 PM. Security analysis identified 4 employees with sufficient system privileges to access the compromised database.

**Question:**

Based on the available evidence, who is most likely responsible for the data breach?

**Hypotheses:**

1. Finley Brooks is responsible
2. Indigo Taylor is responsible
3. Casey Morgan is responsible
4. Grey Martinez is responsible
5. Cannot be determined from available evidence ✓

**Evidence (7 items):**

- **E001** [low]: Network analysis shows the breach originated from the shared network segment used by all four suspects' workstations, but the specific machine cannot be identified.
  - *Supports:* Finley Brooks, Indigo Taylor, Casey Morgan, Grey Martinez
- **E002** [low]: Database access logs show all four suspects (Finley Brooks, Indigo Taylor, Casey Morgan, Grey Martinez) accessed the compromised database within the relevant time window as part of their normal duties.
  - *Supports:* Finley Brooks, Indigo Taylor, Casey Morgan, Grey Martinez
- **E003** [low]: Finley Brooks has sufficient database privileges to extract the stolen data and was logged in during the breach window around 8:15 PM.
  - *Supports:* Finley Brooks
- **E004** [low]: Indigo Taylor has sufficient database privileges to extract the stolen data and was logged in during the breach window around 8:30 PM.
  - *Supports:* Indigo Taylor
- **E005** [low]: Casey Morgan has sufficient database privileges to extract the stolen data and was logged in during the breach window around 8:45 PM.
  - *Supports:* Casey Morgan
- **E006** [low]: Grey Martinez has sufficient database privileges to extract the stolen data and was logged in during the breach window around 9:00 PM.
  - *Supports:* Grey Martinez
- **E007** [none]: Detailed forensic logs that could identify the responsible party were not enabled on the database server at the time of the breach.

**Gold Answer:**

Cannot be determined from available evidence

**Gold Reasoning:**

The evidence is genuinely ambiguous and insufficient to identify a specific responsible party. All suspects had equal access, opportunity, and capability. No diagnostic evidence uniquely implicates any individual suspect, and critical forensic data is either unavailable or non-discriminating. A definitive determination cannot be made without additional evidence.

---

### Item 4: t2_insufficient_contamination_015

**Template:** contamination

**Narrative:**

At BioResearch Labs's Building D - Storage Facility, a batch of products was contaminated on March 15, 2024, requiring a costly recall. Quality control investigation narrowed the incident to 4 staff members who handled the batch during production.

**Question:**

Based on the available evidence, who is most likely responsible for the contamination?

**Hypotheses:**

1. Blake Rivera is responsible
2. Ellis Kim is responsible
3. Grey Martinez is responsible
4. Finley Brooks is responsible
5. Cannot be determined from available evidence ✓

**Evidence (7 items):**

- **E001** [low]: Production records show the contaminated batch was processed through multiple stations, with all four suspects (Blake Rivera, Ellis Kim, Grey Martinez, Finley Brooks) handling it at different stages.
  - *Supports:* Blake Rivera, Ellis Kim, Grey Martinez, Finley Brooks
- **E002** [low]: Blake Rivera worked on the batch during their assigned shift around 8:15 PM and had access to materials that could cause the observed contamination.
  - *Supports:* Blake Rivera
- **E003** [low]: Ellis Kim worked on the batch during their assigned shift around 8:30 PM and had access to materials that could cause the observed contamination.
  - *Supports:* Ellis Kim
- **E004** [low]: Grey Martinez worked on the batch during their assigned shift around 8:45 PM and had access to materials that could cause the observed contamination.
  - *Supports:* Grey Martinez
- **E005** [low]: Finley Brooks worked on the batch during their assigned shift around 9:00 PM and had access to materials that could cause the observed contamination.
  - *Supports:* Finley Brooks
- **E006** [low]: Testing of equipment and work areas found trace amounts of the contaminant at multiple stations where all four suspects worked.
  - *Supports:* Blake Rivera, Ellis Kim, Grey Martinez, Finley Brooks
- **E007** [none]: The contamination could have occurred at any of the four processing stages, and there is no physical evidence to determine which stage was the source.

**Gold Answer:**

Cannot be determined from available evidence

**Gold Reasoning:**

The evidence is genuinely ambiguous and insufficient to identify a specific responsible party. All suspects had equal access, opportunity, and capability. No diagnostic evidence uniquely implicates any individual suspect, and critical forensic data is either unavailable or non-discriminating. A definitive determination cannot be made without additional evidence.

---


